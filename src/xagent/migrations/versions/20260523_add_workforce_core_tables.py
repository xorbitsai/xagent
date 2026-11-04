"""add workforce core tables

Revision ID: 20260523_add_workforce_core_tables
Revises: 20260522_add_task_chat_message_turn_id
Create Date: 2026-05-23 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260523_add_workforce_core_tables"
down_revision: Union[str, None] = "20260522_add_task_chat_message_turn_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _inspector() -> sa.Inspector:
    from alembic import context

    return sa.inspect(context.get_bind())


def _table_exists(table_name: str) -> bool:
    return table_name in _inspector().get_table_names()


def _index_exists(table_name: str, index_name: str) -> bool:
    if not _table_exists(table_name):
        return False
    return index_name in {idx["name"] for idx in _inspector().get_indexes(table_name)}


def _create_index_if_missing(
    index_name: str,
    table_name: str,
    columns: list[str],
    *,
    unique: bool = False,
) -> None:
    if not _index_exists(table_name, index_name):
        op.create_index(op.f(index_name), table_name, columns, unique=unique)


def _drop_index_if_exists(index_name: str, table_name: str) -> None:
    if _index_exists(table_name, index_name):
        op.drop_index(op.f(index_name), table_name=table_name)


def _foreign_key_if_table_exists(
    table_name: str,
    columns: list[str],
    referent: list[str],
    *,
    ondelete: str,
) -> list[sa.ForeignKeyConstraint]:
    if not _table_exists(table_name):
        return []
    return [sa.ForeignKeyConstraint(columns, referent, ondelete=ondelete)]


def upgrade() -> None:
    if not _table_exists("workforces"):
        op.create_table(
            "workforces",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("owner_user_id", sa.Integer(), nullable=False),
            sa.Column("scope_type", sa.String(length=50), nullable=False),
            sa.Column("scope_id", sa.String(length=200), nullable=False),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("manager_agent_id", sa.Integer(), nullable=False),
            sa.Column("manager_instructions", sa.Text(), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("canvas_layout", sa.JSON(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            *_foreign_key_if_table_exists(
                "agents",
                ["manager_agent_id"],
                ["agents.id"],
                ondelete="RESTRICT",
            ),
            *_foreign_key_if_table_exists(
                "users",
                ["owner_user_id"],
                ["users.id"],
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "scope_type", "scope_id", "name", name="uq_workforce_scope_name"
            ),
        )

    _create_index_if_missing("ix_workforces_id", "workforces", ["id"])
    _create_index_if_missing(
        "ix_workforces_manager_agent_id", "workforces", ["manager_agent_id"]
    )
    _create_index_if_missing(
        "ix_workforces_owner_user_id", "workforces", ["owner_user_id"]
    )
    _create_index_if_missing("ix_workforces_scope_id", "workforces", ["scope_id"])
    _create_index_if_missing("ix_workforces_scope_type", "workforces", ["scope_type"])
    _create_index_if_missing("ix_workforces_status", "workforces", ["status"])

    if not _table_exists("workforce_agents"):
        op.create_table(
            "workforce_agents",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("workforce_id", sa.Integer(), nullable=False),
            sa.Column("agent_id", sa.Integer(), nullable=False),
            sa.Column("alias", sa.String(length=200), nullable=True),
            sa.Column("assignment_instructions", sa.Text(), nullable=False),
            sa.Column("source_type", sa.String(length=20), nullable=False),
            sa.Column("template_id", sa.String(length=200), nullable=True),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("sort_order", sa.Integer(), nullable=False),
            sa.Column("canvas_position", sa.JSON(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            *_foreign_key_if_table_exists(
                "agents",
                ["agent_id"],
                ["agents.id"],
                ondelete="RESTRICT",
            ),
            *_foreign_key_if_table_exists(
                "workforces",
                ["workforce_id"],
                ["workforces.id"],
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("workforce_id", "agent_id", name="uq_workforce_agent"),
        )

    _create_index_if_missing(
        "ix_workforce_agents_agent_id", "workforce_agents", ["agent_id"]
    )
    _create_index_if_missing("ix_workforce_agents_id", "workforce_agents", ["id"])
    _create_index_if_missing(
        "ix_workforce_agents_workforce_id", "workforce_agents", ["workforce_id"]
    )

    if not _table_exists("workforce_runs"):
        op.create_table(
            "workforce_runs",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("workforce_id", sa.Integer(), nullable=False),
            sa.Column("task_id", sa.Integer(), nullable=True),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("snapshot", sa.JSON(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            *_foreign_key_if_table_exists(
                "tasks",
                ["task_id"],
                ["tasks.id"],
                ondelete="SET NULL",
            ),
            *_foreign_key_if_table_exists(
                "users",
                ["user_id"],
                ["users.id"],
                ondelete="CASCADE",
            ),
            *_foreign_key_if_table_exists(
                "workforces",
                ["workforce_id"],
                ["workforces.id"],
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("task_id"),
        )

    _create_index_if_missing("ix_workforce_runs_id", "workforce_runs", ["id"])
    _create_index_if_missing("ix_workforce_runs_status", "workforce_runs", ["status"])
    _create_index_if_missing("ix_workforce_runs_user_id", "workforce_runs", ["user_id"])
    _create_index_if_missing(
        "ix_workforce_runs_workforce_id", "workforce_runs", ["workforce_id"]
    )

    if not _table_exists("workforce_builder_messages"):
        op.create_table(
            "workforce_builder_messages",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("workforce_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("role", sa.String(length=20), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("proposed_patch", sa.JSON(), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            *_foreign_key_if_table_exists(
                "users",
                ["user_id"],
                ["users.id"],
                ondelete="CASCADE",
            ),
            *_foreign_key_if_table_exists(
                "workforces",
                ["workforce_id"],
                ["workforces.id"],
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    _create_index_if_missing(
        "ix_workforce_builder_messages_id", "workforce_builder_messages", ["id"]
    )
    _create_index_if_missing(
        "ix_workforce_builder_messages_user_id",
        "workforce_builder_messages",
        ["user_id"],
    )
    _create_index_if_missing(
        "ix_workforce_builder_messages_workforce_id",
        "workforce_builder_messages",
        ["workforce_id"],
    )


def downgrade() -> None:
    _drop_index_if_exists(
        "ix_workforce_builder_messages_workforce_id",
        "workforce_builder_messages",
    )
    _drop_index_if_exists(
        "ix_workforce_builder_messages_user_id", "workforce_builder_messages"
    )
    _drop_index_if_exists(
        "ix_workforce_builder_messages_id", "workforce_builder_messages"
    )
    if _table_exists("workforce_builder_messages"):
        op.drop_table("workforce_builder_messages")

    _drop_index_if_exists("ix_workforce_runs_workforce_id", "workforce_runs")
    _drop_index_if_exists("ix_workforce_runs_user_id", "workforce_runs")
    _drop_index_if_exists("ix_workforce_runs_status", "workforce_runs")
    _drop_index_if_exists("ix_workforce_runs_id", "workforce_runs")
    if _table_exists("workforce_runs"):
        op.drop_table("workforce_runs")

    _drop_index_if_exists("ix_workforce_agents_workforce_id", "workforce_agents")
    _drop_index_if_exists("ix_workforce_agents_id", "workforce_agents")
    _drop_index_if_exists("ix_workforce_agents_agent_id", "workforce_agents")
    if _table_exists("workforce_agents"):
        op.drop_table("workforce_agents")

    _drop_index_if_exists("ix_workforces_status", "workforces")
    _drop_index_if_exists("ix_workforces_scope_type", "workforces")
    _drop_index_if_exists("ix_workforces_scope_id", "workforces")
    _drop_index_if_exists("ix_workforces_owner_user_id", "workforces")
    _drop_index_if_exists("ix_workforces_manager_agent_id", "workforces")
    _drop_index_if_exists("ix_workforces_id", "workforces")
    if _table_exists("workforces"):
        op.drop_table("workforces")
