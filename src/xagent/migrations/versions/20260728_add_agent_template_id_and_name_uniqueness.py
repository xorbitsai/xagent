"""Add agents.template_id and per-user unique-name / quick-access constraints.

``template_id`` lets the create-or-reuse-from-template flow key off a
stable id instead of the user-editable display name. The unique index on
(user_id, name) backs the app-level ``agent_name_exists`` check at the
database layer for a single user's own check-then-insert race; it excludes
``workforce_generated_manager`` agents to match ``agent_name_exists``,
which already deliberately allows those to share names. Note this index is
strictly per-``user_id`` - when a SaaS team-scope hook is installed,
``agent_name_exists`` becomes a team-wide check that this index does not
mirror (see the docstring on ``agent_team_scope.owned_agent_clause`` and
the comment on ``Agent.__table_args__``).

A second partial unique index on (user_id, template_id), scoped to the
``template_quick_access`` origin, backs the /task template quick-access
resolve flow's get-or-create atomicity (PR review findings B1/B2/D2/D3): a
plain ``origin != 'template_quick_access'`` create (e.g. the
workforce-builder UI's ``POST /from-template``, which deliberately mints
multiple named instances of one template) is untouched by it. No rows carry
this origin before this migration ships, so unlike the name index above,
this one needs no pre-index dedupe pass.

Existing duplicate (user_id, name) rows (if any) are renamed before the
name index is created so this migration cannot fail on already-messy data.
Renaming is a one-way, best-effort disambiguation: ``downgrade()`` drops the
column/indexes but does not attempt to restore the original colliding
names, since which row "owned" the pre-migration name is no longer
recoverable once other rows have shifted around it.

The dedupe pass and index creation both run inside this migration's single
transaction, so any lock the database takes for either (e.g. a table-level
lock while building a partial index, on backends that need one) is held for
the combined duration of both steps, not just the index build. Expected to
be negligible for typical agent-table sizes; revisit if this ever runs
against a very large, highly-contended ``agents`` table in production.

Revision ID: 20260728_add_agent_template_id_and_name_uniqueness
Revises: 20260801_add_trigger_consecutive_prepare_failures
Create Date: 2026-07-28
"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

revision: str = "20260728_add_agent_template_id_and_name_uniqueness"
down_revision: Union[str, None] = "20260801_add_trigger_consecutive_prepare_failures"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "agents"
TEMPLATE_ID_COLUMN = "template_id"
TEMPLATE_ID_INDEX = "ix_agents_template_id"
NAME_UNIQUE_INDEX = "uq_agents_user_id_name_active"
TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX = "uq_agents_user_id_template_id_quick_access"
WORKFORCE_MANAGER_ORIGIN = "workforce_generated_manager"
QUICK_ACCESS_ORIGIN = "template_quick_access"
# Agent.name is String(200) - renamed candidates must fit within that.
MAX_NAME_LENGTH = 200
# Matches Agent.__table_args__'s _NON_WORKFORCE_MANAGER_CLAUSE /
# _QUICK_ACCESS_ORIGIN_CLAUSE (agent.py): built once at module scope rather
# than re-interpolated per call, so this migrations directory doesn't pick
# up per-call f-string DDL predicates as a pattern (both origin constants
# are fixed module constants, not user input, so this was never an
# injection risk).
NAME_UNIQUE_INDEX_WHERE_CLAUSE = sa.text(f"origin != '{WORKFORCE_MANAGER_ORIGIN}'")
TEMPLATE_QUICK_ACCESS_WHERE_CLAUSE = sa.text(f"origin = '{QUICK_ACCESS_ORIGIN}'")


def _table_names() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _column_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return {column["name"] for column in inspector.get_columns(table_name)}


def _index_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return {
        name
        for item in inspector.get_indexes(table_name)
        if (name := item.get("name")) is not None
    }


def _fit_to_column(base: str, suffix: str) -> str:
    """Truncate ``base`` so ``base + suffix`` fits ``Agent.name``'s
    String(200) column. A long pre-existing name plus the rename suffix
    could otherwise overflow and abort the migration on backends (e.g.
    Postgres) that enforce column length at insert/update time - the exact
    data this migration exists to survive.
    """
    available = MAX_NAME_LENGTH - len(suffix)
    if available <= 0:
        # Pathological (suffix alone doesn't fit) - truncate the suffix
        # itself as a last resort rather than raise a negative-slice error.
        return suffix[:MAX_NAME_LENGTH]
    return base[:available] + suffix


def _dedupe_agent_names() -> None:
    """Rename losing rows of any existing (user_id, name) duplicate group.

    Only considers non-workforce-manager agents, matching the partial index's
    scope. The lowest ``id`` in each group keeps its name; later rows are
    suffixed with their own id, mirroring the disambiguation the frontend
    already applies on a name collision. If that suffixed name happens to
    already belong to another row for the same user (e.g. an existing agent
    literally named ``"Foo (123)"``), an incrementing counter is appended
    until a genuinely free name is found, so the rename itself can never
    introduce a *new* collision for the unique index we're about to build.
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

    # Each affected user's existing names are fetched once and kept updated
    # locally as renames are chosen, instead of issuing one SELECT per
    # candidate-suffix attempt (this also catches a collision against another
    # duplicate group for the same user, not just the current group).
    taken_names_by_user: dict[int, set[str]] = {}

    def _taken_names(user_id: int) -> set[str]:
        if user_id not in taken_names_by_user:
            rows = bind.execute(
                sa.select(agents.c.name).where(
                    agents.c.user_id == user_id,
                    agents.c.origin != WORKFORCE_MANAGER_ORIGIN,
                )
            ).fetchall()
            taken_names_by_user[user_id] = {row[0] for row in rows}
        return taken_names_by_user[user_id]

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

        taken = _taken_names(user_id)

        for (agent_id,) in rows[1:]:
            candidate = _fit_to_column(name, f" ({agent_id})")
            suffix = 2
            while candidate in taken:
                candidate = _fit_to_column(name, f" ({agent_id}-{suffix})")
                suffix += 1

            # The rename is one-way (downgrade doesn't restore it - see the
            # module docstring), so leave an audit trail of exactly which
            # rows were mutated instead of renaming silently.
            logger.warning(
                "Deduplicating agent name before unique-index creation: "
                "agent id=%s (user_id=%s) renamed %r -> %r",
                agent_id,
                user_id,
                name,
                candidate,
            )
            bind.execute(
                sa.update(agents).where(agents.c.id == agent_id).values(name=candidate)
            )
            taken.add(candidate)


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
        op.create_index(
            NAME_UNIQUE_INDEX,
            TABLE,
            ["user_id", "name"],
            unique=True,
            sqlite_where=NAME_UNIQUE_INDEX_WHERE_CLAUSE,
            postgresql_where=NAME_UNIQUE_INDEX_WHERE_CLAUSE,
        )

    if TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX not in _index_names(TABLE):
        # No dedupe pass needed here: the quick-access origin is brand new
        # as of this migration, so no pre-existing row can carry it.
        op.create_index(
            TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX,
            TABLE,
            ["user_id", "template_id"],
            unique=True,
            sqlite_where=TEMPLATE_QUICK_ACCESS_WHERE_CLAUSE,
            postgresql_where=TEMPLATE_QUICK_ACCESS_WHERE_CLAUSE,
        )


def downgrade() -> None:
    if TABLE not in _table_names():
        return

    if NAME_UNIQUE_INDEX in _index_names(TABLE):
        op.drop_index(NAME_UNIQUE_INDEX, table_name=TABLE)

    if TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX in _index_names(TABLE):
        op.drop_index(TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX, table_name=TABLE)

    if TEMPLATE_ID_INDEX in _index_names(TABLE):
        op.drop_index(TEMPLATE_ID_INDEX, table_name=TABLE)
    if TEMPLATE_ID_COLUMN in _column_names(TABLE):
        op.drop_column(TABLE, TEMPLATE_ID_COLUMN)
