"""add partial index on task_execution_commands for the oldest claimable command

The dispatcher's fallback claim lookup takes the oldest ``id`` among commands
that are still ``pending`` or ``processing``, of any kind. The ``(kind, id)``
claimable index cannot order that query on PostgreSQL 17, which has no skip
scan, so with an empty queue it walks the whole primary key across every
settled row. A partial index on ``id`` scoped to those statuses gives it id
order over only the claimable rows.

On PostgreSQL the index is built ``CONCURRENTLY`` in ``autocommit_block()`` so
writes are not blocked, and an ``INVALID`` index or one whose columns or
predicate differ is rebuilt. SQLite gets the same partial index through
``sqlite_where``, built without ``CONCURRENTLY``. Any other dialect gets a
plain ``CREATE INDEX`` without a predicate.

Accepted cost: every insert, and every status change that leaves a row
``pending`` or ``processing``, maintains one more index entry.

Revision ID: 20261015_task_cmd_claimable_id
Revises: 20261014_encrypt_user_oauth_tokens
Create Date: 2026-10-15

"""

import re
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261015_task_cmd_claimable_id"
down_revision: Union[str, None] = "20261014_encrypt_user_oauth_tokens"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "task_execution_commands"
INDEX = "ix_task_commands_claimable_id"
INDEX_COLUMNS = ("id",)
# The predicate column itself, so a table that lacks ``status`` is recognized
# as not ready for this index, same as the two index columns.
REQUIRED_COLUMNS = (*INDEX_COLUMNS, "status")
PREDICATE = sa.text("status IN ('pending', 'processing')")

# A same-named (id) index with a stale or missing partial predicate has
# the same column list as a correct one, so the deparsed predicate is read too.
POSTGRES_INDEX_STATE_SQL = sa.text(
    """
    SELECT i.indisvalid, pg_catalog.pg_get_expr(i.indpred, i.indrelid)
    FROM pg_catalog.pg_index AS i
    WHERE i.indexrelid = pg_catalog.to_regclass(:index_name)
    """
)


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _columns(inspector: sa.Inspector) -> set[str]:
    if TABLE not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(TABLE)}


def _indexes(inspector: sa.Inspector) -> set[str]:
    if TABLE not in inspector.get_table_names():
        return set()
    return {
        name
        for item in inspector.get_indexes(TABLE)
        if (name := item.get("name")) is not None
    }


def _index_columns(inspector: sa.Inspector, index_name: str) -> tuple[str, ...] | None:
    if TABLE not in inspector.get_table_names():
        return None
    for item in inspector.get_indexes(TABLE):
        if item.get("name") == index_name:
            return tuple(str(name) for name in item.get("column_names") or ())
    return None


def _postgres_index_state() -> tuple[bool, str | None] | None:
    """Return (is valid, deparsed predicate) of the index, or None if absent."""

    row = (
        op.get_bind()
        .execute(POSTGRES_INDEX_STATE_SQL, {"index_name": INDEX})
        .one_or_none()
    )
    return None if row is None else (bool(row[0]), row[1])


def _normalize_pg_predicate(expression: str) -> str:
    """Normalize a Postgres-deparsed predicate for structural comparison.

    ``pg_get_expr()`` reformats the predicate it stored: it wraps the column
    reference in parentheses, casts the column and the literals to the
    operator's resolved type, rewrites an ``IN`` list into
    ``= ANY (ARRAY[...])`` (this migration's own predicate comes back as
    ``((status)::text = ANY ((ARRAY['pending'::character varying,
    'processing'::character varying])::text[]))``), and does not guarantee
    whitespace. None of that changes what the predicate selects, so the
    comparison removes casts, parentheses and whitespace, restores an
    ``= ANY (ARRAY[...])`` list to ``IN``, and compares the whole string.
    """

    normalized = expression.lower()
    normalized = re.sub(r"::[a-z_ ]+(\([0-9]+\))?(\[\])?", "", normalized)
    normalized = re.sub(r"[()\s]", "", normalized)
    return re.sub(r"=anyarray\[(.*)\]$", r"in\1", normalized)


def _postgres_predicate_matches_expected(expression: str | None) -> bool:
    """Whether ``expression`` is our own PREDICATE (a non-partial index,
    ``expression is None``, never matches)."""

    if expression is None:
        return False
    return _normalize_pg_predicate(expression) == _normalize_pg_predicate(
        str(PREDICATE)
    )


def _upgrade_postgresql() -> None:
    """Build the claimable-id index without blocking table writes."""

    inspector = _inspector()
    if not set(REQUIRED_COLUMNS) <= _columns(inspector):
        return

    state = _postgres_index_state()
    existing_columns = _index_columns(inspector, INDEX)
    if (
        state is not None
        and state[0]
        and existing_columns == INDEX_COLUMNS
        and _postgres_predicate_matches_expected(state[1])
    ):
        return

    with op.get_context().autocommit_block():
        if state is not None or existing_columns is not None:
            op.drop_index(
                INDEX,
                table_name=TABLE,
                if_exists=True,
                postgresql_concurrently=True,
            )
        op.create_index(
            INDEX,
            TABLE,
            list(INDEX_COLUMNS),
            if_not_exists=True,
            postgresql_concurrently=True,
            postgresql_where=PREDICATE,
        )


def upgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            with context.autocommit_block():
                op.create_index(
                    INDEX,
                    TABLE,
                    list(INDEX_COLUMNS),
                    postgresql_concurrently=True,
                    postgresql_where=PREDICATE,
                )
        elif context.dialect.name == "sqlite":
            op.create_index(INDEX, TABLE, list(INDEX_COLUMNS), sqlite_where=PREDICATE)
        else:
            op.create_index(INDEX, TABLE, list(INDEX_COLUMNS))
        return

    if context.dialect.name == "postgresql":
        _upgrade_postgresql()
        return

    inspector = _inspector()
    # A database whose table lacks one of these columns (an Alembic-only
    # install stopped short of the migration that added it, or a legacy
    # fixture in a test) cannot support this predicate or these index columns
    # yet; skip rather than fail the whole chain.
    if not set(REQUIRED_COLUMNS) <= _columns(inspector):
        return
    if INDEX in _indexes(inspector):
        return
    kwargs: dict[str, object] = {}
    if context.dialect.name == "sqlite":
        kwargs["sqlite_where"] = PREDICATE
    op.create_index(INDEX, TABLE, list(INDEX_COLUMNS), **kwargs)


def downgrade() -> None:
    context = op.get_context()
    if context.dialect.name == "postgresql":
        with context.autocommit_block():
            op.drop_index(
                INDEX,
                table_name=TABLE,
                if_exists=not context.as_sql,
                postgresql_concurrently=True,
            )
    elif context.as_sql or INDEX in _indexes(_inspector()):
        op.drop_index(INDEX, table_name=TABLE)
