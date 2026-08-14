"""add tasks.interaction_protocol_version

Revision ID: 20260810_add_task_interaction_protocol_version
Revises: 20260809_add_task_interaction_requests
Create Date: 2026-08-10

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260810_add_task_interaction_protocol_version"
down_revision: Union[str, None] = "20260809_add_task_interaction_requests"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "tasks"
COLUMN = "interaction_protocol_version"
CONSTRAINT_NAME = "ck_tasks_interaction_protocol_version"
CONSTRAINT_CONDITION = (
    "interaction_protocol_version IS NULL OR interaction_protocol_version = 1"
)

# The schema of the *visible* tasks relation. version_table_schema names only
# the Alembic version table, and current_schema() is merely the first entry on
# search_path, so neither identifies the relation an unqualified reference
# actually resolves to. Ask PostgreSQL which one it resolves.
POSTGRES_VISIBLE_TABLE_SCHEMA_SQL = sa.text(
    """
    SELECT ns.nspname
    FROM pg_catalog.pg_class AS cls
    JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
    WHERE cls.oid = pg_catalog.to_regclass(:table_name)
    """
)


def _target_schema() -> str | None:
    """The schema holding the tasks relation this migration operates on."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        resolved = bind.execute(
            POSTGRES_VISIBLE_TABLE_SCHEMA_SQL, {"table_name": TABLE}
        ).scalar()
        if resolved:
            return str(resolved)
    schema = op.get_context().version_table_schema
    return str(schema) if schema else None


def _online_inventory(schema: str | None) -> tuple[set[str], set[str]] | None:
    """One inspector, one table-names fetch: the tasks columns and the names
    of its ck_ CHECK constraints, or None when tasks does not exist. No
    branch in this migration creates or drops tasks -- only columns and
    constraints on it -- so a single fetch cannot go stale. Do not reuse the
    returned sets after a DDL statement: they are a snapshot, and the
    inspector behind them caches.
    """
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names(schema=schema):
        return None
    columns = {
        str(item["name"]) for item in inspector.get_columns(TABLE, schema=schema)
    }
    checks = {
        str(item["name"])
        for item in inspector.get_check_constraints(TABLE, schema=schema)
        if item.get("name")
    }
    return columns, checks


def upgrade() -> None:
    context = op.get_context()

    # Offline (--sql) generation has a MockConnection, so reflection is
    # unavailable. Emit the unconditional DDL instead of inspecting. Adding
    # a CHECK to an existing table has no --sql-mode-safe path on SQLite --
    # a batch_alter_table rebuild that could add it cannot render under
    # --sql mode on either dialect, and offline SQL support is a hard
    # requirement -- so only PostgreSQL gets the constraint on this branch.
    # The ONLINE branch below does converge SQLite, via exactly that
    # rebuild; only offline-generated SQLite upgrade scripts still produce
    # the unconstrained shape. See
    # test_sqlite_check_matches_between_migration_and_create_all in
    # tests/migrations/test_task_interaction_protocol_version_parity.py.
    if context.as_sql:
        op.add_column(TABLE, sa.Column(COLUMN, sa.Integer(), nullable=True))
        if context.dialect.name == "postgresql":
            op.create_check_constraint(CONSTRAINT_NAME, TABLE, CONSTRAINT_CONDITION)
        return

    # Address the same relation the catalog lookup inspects, so reflection and
    # DDL can never diverge onto different schemas.
    schema = _target_schema()

    inventory = _online_inventory(schema)
    if inventory is None:
        return
    columns, checks = inventory

    if COLUMN not in columns:
        op.add_column(
            TABLE,
            sa.Column(COLUMN, sa.Integer(), nullable=True),
            schema=schema,
        )

    # The column guard above can skip add_column (e.g. a create_all-built
    # database already has the column), but the CHECK must still be checked
    # independently -- otherwise upgrading such a database would silently
    # leave it without the constraint. ``checks`` was read before the
    # add_column above and is still valid here: adding a column creates no
    # CHECK constraint.
    if context.dialect.name == "postgresql":
        # On PostgreSQL create_check_constraint takes an ACCESS EXCLUSIVE
        # lock on tasks and validates every existing row before returning
        # (see 20260804_add_task_checkpoint_trace_event_anchor.py for the
        # same note against create_foreign_key). Here the column was added
        # moments earlier in this same migration, so every row is still
        # NULL and validation cannot fail -- but on a large deployment, a
        # NOT VALID constraint followed by a separate VALIDATE CONSTRAINT
        # would move validation out of the lock window if that ever becomes
        # a concern.
        if CONSTRAINT_NAME not in checks:
            op.create_check_constraint(
                CONSTRAINT_NAME, TABLE, CONSTRAINT_CONDITION, schema=schema
            )
        return

    # SQLite: a plain create_check_constraint raises NotImplementedError,
    # so the CHECK is added by rebuilding the table, the mirror image of
    # what downgrade() below already does to remove it. Without this branch
    # the two SQLite install paths enforce different invariants on the same
    # column at the same version: a fresh install (stamp head, then
    # create_all) carries the CHECK, a database that walked the revision
    # chain does not, and interaction_protocol_version = 2 is rejected on
    # one and accepted on the other. The offline (--sql) branch above
    # cannot converge -- batch mode has no rendering under --sql on either
    # dialect -- so an offline-generated SQLite upgrade still produces the
    # unconstrained shape; that residue is the one asymmetry this change
    # does not close, and it is asserted rather than left uncovered.
    #
    # The rebuild copies existing rows through the new CHECK. Every row's
    # value is NULL or 1 today (this column has no writer that emits
    # anything else), so the copy cannot fail; a deployment that somehow
    # holds another value must fix the data before upgrading, which is the
    # loud failure this constraint exists to produce.
    if CONSTRAINT_NAME not in checks:
        with op.batch_alter_table(TABLE, schema=schema) as batch_op:
            batch_op.create_check_constraint(CONSTRAINT_NAME, CONSTRAINT_CONDITION)


def downgrade() -> None:
    context = op.get_context()

    # Constraint dropped before column, both here and in the online
    # PostgreSQL branch below. PostgreSQL does not require this order --
    # DROP COLUMN auto-drops a single-column CHECK that depends on the
    # dropped column, no CASCADE needed. The explicit drop_constraint keeps
    # the downgrade symmetric with upgrade's add-column-then-add-constraint
    # sequence and independent of that auto-drop behaviour.
    #
    # The SQLite side of this offline branch stays a plain drop_column,
    # unlike the batch_alter_table rebuild the online branch below uses.
    # Offline (--sql) generation cannot run batch mode, so that rebuild has
    # no offline rendering here -- see
    # 20260804_add_task_checkpoint_trace_event_anchor.py, which takes the
    # same shape on this same table and documents accepting the loud
    # first-statement failure this produces against a create_all-built
    # SQLite database. Fixing this revision alone would not make the
    # offline downgrade chain traversable for that shape either: two
    # revisions further down, 20260804 fails the same way, so the failure
    # is a property of the repository's offline-downgrade support, not of
    # this revision. The alternative -- an offline batch_alter_table
    # rebuild via copy_from=<a frozen Table object> -- needs a full table
    # definition frozen inside the migration; if that frozen copy ever
    # drifts from the real table, the rendered rebuild silently drops the
    # missing columns' data instead of failing, which is strictly worse
    # than today's loud first-statement error. Downgrading a
    # create_all-built SQLite database over a live connection takes the
    # rebuilding branch below instead, which reflects the real table and
    # needs no frozen copy.
    if context.as_sql:
        if context.dialect.name == "postgresql":
            op.drop_constraint(CONSTRAINT_NAME, TABLE, type_="check")
        op.drop_column(TABLE, COLUMN)
        return

    schema = _target_schema()

    inventory = _online_inventory(schema)
    if inventory is None:
        return
    columns, checks = inventory

    if context.dialect.name == "postgresql":
        if CONSTRAINT_NAME in checks:
            op.drop_constraint(CONSTRAINT_NAME, TABLE, type_="check", schema=schema)

        if COLUMN in columns:
            op.drop_column(TABLE, COLUMN, schema=schema)
        return

    # Non-PostgreSQL (SQLite): both installation shapes now carry
    # ck_tasks_interaction_protocol_version -- a fresh install via
    # create_all (see src/xagent/db/migration.py's empty-DB stamp path and
    # src/xagent/web/models/database.py's create_all call, at its
    # _initialize_database_schema site), and one that walked upgrade()
    # above, which now rebuilds the table to add the CHECK the same way
    # this branch rebuilds it to remove one (see
    # test_sqlite_check_matches_between_migration_and_create_all in
    # tests/migrations/test_task_interaction_protocol_version_parity.py).
    # SQLite 3.35+ refuses to DROP a column a CHECK constraint references, so
    # a plain drop_column() breaks on that shape. batch_alter_table rebuilds
    # the table (copy/rename/drop) instead, which drops the constraint and
    # the column together regardless of which shape this database has.
    if COLUMN in columns:
        constraint_present = CONSTRAINT_NAME in checks
        with op.batch_alter_table(TABLE, schema=schema) as batch_op:
            if constraint_present:
                batch_op.drop_constraint(CONSTRAINT_NAME, type_="check")
            batch_op.drop_column(COLUMN)
