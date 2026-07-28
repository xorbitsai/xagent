"""Add agents.template_id and a per-user unique name constraint.

``template_id`` lets the create-or-reuse-from-template flow key off a
stable id instead of the user-editable display name. The unique index on
(user_id, name) backs the existing app-level ``agent_name_exists`` check at
the database layer, closing the check-then-insert race; it excludes
``workforce_generated_manager`` agents to match ``agent_name_exists``,
which already deliberately allows those to share names.

Existing duplicate (user_id, name) rows (if any) are renamed before the
index is created so this migration cannot fail on already-messy data.

Revision ID: 20260728_add_agent_template_id_and_name_uniqueness
Revises: 20260725_add_uploaded_file_recovery_index
Create Date: 2026-07-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260728_add_agent_template_id_and_name_uniqueness"
down_revision: Union[str, None] = "20260725_add_uploaded_file_recovery_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "agents"
TEMPLATE_ID_COLUMN = "template_id"
TEMPLATE_ID_INDEX = "ix_agents_template_id"
NAME_UNIQUE_INDEX = "uq_agents_user_id_name_active"
WORKFORCE_MANAGER_ORIGIN = "workforce_generated_manager"


def _table_names() -> set[str]:
    return set(Inspector.from_engine(op.get_bind()).get_table_names())


def _column_names(table_name: str) -> set[str]:
    inspector = Inspector.from_engine(op.get_bind())
    return {column["name"] for column in inspector.get_columns(table_name)}


def _index_names(table_name: str) -> set[str]:
    inspector = Inspector.from_engine(op.get_bind())
    return {
        name
        for item in inspector.get_indexes(table_name)
        if (name := item.get("name")) is not None
    }


def _dedupe_agent_names() -> None:
    """Rename losing rows of any existing (user_id, name) duplicate group.

    Only considers non-workforce-manager agents, matching the partial index's
    scope. The lowest ``id`` in each group keeps its name; later rows are
    suffixed with their own id, mirroring the disambiguation the frontend
    already applies on a name collision.
    """
    bind = op.get_bind()
    agents = sa.table(
        TABLE,
        sa.column("id", sa.Integer),
        sa.column("user_id", sa.Integer),
        sa.column("name", sa.String),
        sa.column("origin", sa.String),
    )

    duplicate_groups = bind.execute(
        sa.select(agents.c.user_id, agents.c.name)
        .where(agents.c.origin != WORKFORCE_MANAGER_ORIGIN)
        .group_by(agents.c.user_id, agents.c.name)
        .having(sa.func.count(agents.c.id) > 1)
    ).fetchall()

    for user_id, name in duplicate_groups:
        rows = bind.execute(
            sa.select(agents.c.id)
            .where(
                agents.c.user_id == user_id,
                agents.c.name == name,
                agents.c.origin != WORKFORCE_MANAGER_ORIGIN,
            )
            .order_by(agents.c.id)
        ).fetchall()

        for (agent_id,) in rows[1:]:
            bind.execute(
                sa.update(agents)
                .where(agents.c.id == agent_id)
                .values(name=f"{name} ({agent_id})")
            )


def upgrade() -> None:
    if TABLE not in _table_names():
        return

    if TEMPLATE_ID_COLUMN not in _column_names(TABLE):
        op.add_column(
            TABLE, sa.Column(TEMPLATE_ID_COLUMN, sa.String(255), nullable=True)
        )
    if TEMPLATE_ID_INDEX not in _index_names(TABLE):
        op.create_index(TEMPLATE_ID_INDEX, TABLE, [TEMPLATE_ID_COLUMN])

    if NAME_UNIQUE_INDEX not in _index_names(TABLE):
        _dedupe_agent_names()
        where_clause = sa.text(f"origin != '{WORKFORCE_MANAGER_ORIGIN}'")
        op.create_index(
            NAME_UNIQUE_INDEX,
            TABLE,
            ["user_id", "name"],
            unique=True,
            sqlite_where=where_clause,
            postgresql_where=where_clause,
        )


def downgrade() -> None:
    if TABLE not in _table_names():
        return

    if NAME_UNIQUE_INDEX in _index_names(TABLE):
        op.drop_index(NAME_UNIQUE_INDEX, table_name=TABLE)

    if TEMPLATE_ID_INDEX in _index_names(TABLE):
        op.drop_index(TEMPLATE_ID_INDEX, table_name=TABLE)
    if TEMPLATE_ID_COLUMN in _column_names(TABLE):
        op.drop_column(TABLE, TEMPLATE_ID_COLUMN)
