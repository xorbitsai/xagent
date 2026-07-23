import logging
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError
from sqlalchemy import Engine, inspect
from sqlalchemy.engine import Connection

from .config import create_alembic_config

logger = logging.getLogger(__name__)

SQLiteForeignKeyDescriptor = tuple[str, tuple[tuple[str, str | None], ...]]
SQLiteScalar = int | float | str | bytes | None
SQLiteRowIdentity = tuple[SQLiteScalar, ...]
SQLiteForeignKeyViolation = tuple[
    str,
    SQLiteRowIdentity,
    SQLiteForeignKeyDescriptor,
]


def _set_sqlite_foreign_keys(connection: Connection, *, enabled: bool) -> None:
    """Set and verify SQLite FK enforcement outside a transaction."""
    if connection.in_transaction():
        connection.rollback()
    expected = 1 if enabled else 0
    connection.exec_driver_sql(f"PRAGMA foreign_keys={expected}")
    connection.commit()
    actual = connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one()
    connection.rollback()
    if actual != expected:
        raise RuntimeError("Could not configure SQLite foreign-key enforcement")


def _sqlite_foreign_key_descriptors(
    connection: Connection,
    table_name: str,
) -> dict[int, SQLiteForeignKeyDescriptor]:
    descriptor_parts: dict[
        int,
        tuple[str, list[tuple[int, str, str | None]]],
    ] = {}
    for (
        constraint_id,
        sequence,
        parent_table,
        child_column,
        parent_column,
    ) in connection.exec_driver_sql(
        'SELECT id, seq, "table", "from", "to" '
        "FROM pragma_foreign_key_list(?) ORDER BY id, seq",
        (table_name,),
    ).all():
        normalized_constraint_id = int(constraint_id)
        if normalized_constraint_id not in descriptor_parts:
            descriptor_parts[normalized_constraint_id] = (str(parent_table), [])
        descriptor_parts[normalized_constraint_id][1].append(
            (
                int(sequence),
                str(child_column),
                str(parent_column) if parent_column is not None else None,
            )
        )
    return {
        constraint_id: (
            parent_table,
            tuple(
                (child_column, parent_column)
                for _sequence, child_column, parent_column in sorted(columns)
            ),
        )
        for constraint_id, (parent_table, columns) in descriptor_parts.items()
    }


def _quote_sqlite_identifier(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def _sqlite_primary_key_columns(
    connection: Connection,
    table_name: str,
) -> tuple[str, ...]:
    return tuple(
        str(column_name)
        for (column_name,) in connection.exec_driver_sql(
            "SELECT name FROM pragma_table_info(?) WHERE pk > 0 ORDER BY pk",
            (table_name,),
        ).all()
    )


def _sqlite_without_rowid_violation_keys(
    connection: Connection,
    table_name: str,
    descriptor: SQLiteForeignKeyDescriptor,
) -> list[SQLiteRowIdentity]:
    primary_key_columns = _sqlite_primary_key_columns(connection, table_name)
    if not primary_key_columns:
        raise RuntimeError("Could not identify SQLite foreign-key violation rows")

    parent_table, column_pairs = descriptor
    parent_primary_key_columns = _sqlite_primary_key_columns(
        connection,
        parent_table,
    )
    if any(parent_column is None for _child_column, parent_column in column_pairs):
        if len(parent_primary_key_columns) != len(column_pairs):
            raise RuntimeError("Could not resolve implicit SQLite foreign-key columns")
        resolved_pairs = tuple(
            (
                child_column,
                parent_column
                if parent_column is not None
                else parent_primary_key_columns[index],
            )
            for index, (child_column, parent_column) in enumerate(column_pairs)
        )
    else:
        resolved_pairs = tuple(
            (child_column, cast(str, parent_column))
            for child_column, parent_column in column_pairs
        )

    quoted_table = _quote_sqlite_identifier(table_name)
    quoted_parent = _quote_sqlite_identifier(parent_table)
    selected_columns = ", ".join(
        f"child.{_quote_sqlite_identifier(column)}" for column in primary_key_columns
    )
    non_null_predicate = " AND ".join(
        f"child.{_quote_sqlite_identifier(child_column)} IS NOT NULL"
        for child_column, _parent_column in resolved_pairs
    )
    parent_match = " AND ".join(
        f"parent.{_quote_sqlite_identifier(parent_column)} = "
        f"child.{_quote_sqlite_identifier(child_column)}"
        for child_column, parent_column in resolved_pairs
    )
    order_by = ", ".join(
        f"child.{_quote_sqlite_identifier(column)}" for column in primary_key_columns
    )
    rows = connection.exec_driver_sql(
        f"SELECT {selected_columns} FROM {quoted_table} AS child "
        f"WHERE {non_null_predicate} AND NOT EXISTS ("
        f"SELECT 1 FROM {quoted_parent} AS parent WHERE {parent_match}"
        f") ORDER BY {order_by}"
    ).all()
    return [cast(SQLiteRowIdentity, tuple(row)) for row in rows]


def _sqlite_foreign_key_violations(
    connection: Connection,
) -> Counter[SQLiteForeignKeyViolation]:
    """Return stable row-and-constraint identities for SQLite FK violations."""
    descriptors_by_table: dict[str, dict[int, SQLiteForeignKeyDescriptor]] = {}
    without_rowid_counts: Counter[tuple[str, SQLiteForeignKeyDescriptor]] = Counter()
    violations: Counter[SQLiteForeignKeyViolation] = Counter()

    for (
        table_name,
        row_id,
        parent_table,
        constraint_id,
    ) in connection.exec_driver_sql("PRAGMA foreign_key_check").all():
        normalized_table_name = str(table_name)
        descriptors = descriptors_by_table.get(normalized_table_name)
        if descriptors is None:
            descriptors = _sqlite_foreign_key_descriptors(
                connection,
                normalized_table_name,
            )
            descriptors_by_table[normalized_table_name] = descriptors
        normalized_constraint_id = int(constraint_id)
        descriptor = descriptors.get(normalized_constraint_id)
        if descriptor is None or descriptor[0] != str(parent_table):
            raise RuntimeError(
                "Could not resolve SQLite foreign-key violation constraint"
            )
        if row_id is None:
            without_rowid_counts[(normalized_table_name, descriptor)] += 1
            continue
        violations[(normalized_table_name, (int(row_id),), descriptor)] += 1

    for (table_name, descriptor), expected_count in without_rowid_counts.items():
        row_keys = _sqlite_without_rowid_violation_keys(
            connection,
            table_name,
            descriptor,
        )
        if not row_keys or expected_count % len(row_keys) != 0:
            raise RuntimeError("Could not identify SQLite foreign-key violation rows")
        multiplicity = expected_count // len(row_keys)
        for row_key in row_keys:
            violations[(table_name, row_key, descriptor)] += multiplicity
    return violations


@contextmanager
def _migration_connection(engine: Engine) -> Iterator[Connection]:
    """Own one migration transaction without SQLite FK cascade side effects."""
    if engine.dialect.name != "sqlite":
        with engine.begin() as connection:
            yield connection
        return

    with engine.connect() as connection:
        foreign_keys_enabled = bool(
            connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one()
        )
        existing_violations = _sqlite_foreign_key_violations(connection)
        connection.rollback()
        if existing_violations:
            logger.warning(
                "SQLite database contains %d pre-existing foreign-key "
                "violation(s); preserving them while rejecting new violations",
                existing_violations.total(),
            )
        try:
            # Alembic batch migrations recreate tables on SQLite. Inbound
            # ON DELETE CASCADE constraints would otherwise treat the temporary
            # parent-table drop as a business deletion and erase child rows.
            _set_sqlite_foreign_keys(connection, enabled=False)
            with connection.begin():
                yield connection
                new_violations = (
                    _sqlite_foreign_key_violations(connection) - existing_violations
                )
                if new_violations:
                    raise RuntimeError(
                        "SQLite migration produced new foreign-key violations"
                    )
        finally:
            _set_sqlite_foreign_keys(
                connection,
                enabled=foreign_keys_enabled,
            )


def is_database_empty(engine: Engine) -> bool:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    return len(tables) == 0


def get_alembic_revision(engine: Engine) -> str | None:
    """Get the current Alembic revision from the database."""
    with engine.connect() as conn:
        context: Any = MigrationContext.configure(conn)
        return cast(str | None, context.get_current_revision())


def _check_revision_is_known(alembic_cfg: Any, engine: Engine, version: str) -> None:
    """Fail with an actionable message when the DB revision is unknown.

    A database whose ``alembic_version`` points at a revision missing from the
    installed package was almost always created or upgraded by a *newer*
    xagent release (e.g. a source checkout or Docker image sharing the same
    storage root). Alembic cannot downgrade without the newer migration
    scripts, so surface what happened and how to recover instead of letting
    ``command.upgrade`` fail with a bare "Can't locate revision" error.
    """
    script = ScriptDirectory.from_config(alembic_cfg)
    try:
        script.get_revisions(version)
    except CommandError:
        db_url = engine.url.render_as_string(hide_password=True)
        raise RuntimeError(
            f"The database at '{db_url}' is at schema revision '{version}', "
            "which this xagent installation does not know about. It was most "
            "likely created or upgraded by a newer version of xagent (for "
            "example a source checkout or Docker image using the same "
            "storage root). To fix this, upgrade xagent to that version or "
            "newer, or point DATABASE_URL / XAGENT_STORAGE_ROOT at a "
            "different database."
        ) from None


def try_upgrade_db(engine: Engine) -> None:
    """Upgrade database to latest migration (or stamp head for brand-new databases)."""
    try:
        logger.info("Starting database upgrade process")
        alembic_cfg = create_alembic_config(engine)
        version = get_alembic_revision(engine)

        # An empty-string version_num (tampered/corrupted alembic_version)
        # is treated like a missing revision: get_revisions("") dies with a
        # bare AssertionError instead of a catchable error.
        if not version:
            if is_database_empty(engine):
                logger.info("Creating new database, stamping to latest revision.")
                with _migration_connection(engine) as conn:
                    alembic_cfg.attributes["connection"] = conn
                    command.stamp(alembic_cfg, "head")
            else:
                raise RuntimeError(
                    "Database exists without alembic revision information. Please initialize the database schema version manually by running: alembic stamp <revision>"
                )
        else:
            _check_revision_is_known(alembic_cfg, engine, version)
            logger.info(f"Current version: {version}, upgrading to head")
            with _migration_connection(engine) as conn:
                alembic_cfg.attributes["connection"] = conn
                command.upgrade(alembic_cfg, "head")
    except Exception as e:
        logger.error(f"Automatic database upgrade failed: {e}")
        raise
