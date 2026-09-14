"""Shared helpers for Alembic migrations that seed rows into shared catalog
tables (e.g. ``public_mcp_apps``) and must avoid clobbering operator data on
downgrade.

These migrations insert a row only if no row with the same id already exists
(so upgrade() never overwrites an operator's pre-existing custom row). Without
a matching guard on the way down, a plain ``DELETE ... WHERE id = :id`` would
delete that operator row anyway, since it only matches on id and does not
check that the row is actually the one the migration inserted. This module is
plain, data-free infrastructure code (not seed data), so migrations may import
it directly.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import sqlalchemy as sa
from sqlalchemy.engine import Connection
from sqlalchemy.sql.expression import TableClause

# Every public_mcp_apps field a pre-existing row can plausibly differ on from
# the migration's seed snapshot, including the two JSON-typed columns
# (oauth_scopes, launch_config): a row that predates a migration's app_id
# being seeded (or predates the admin API's built-in protections) can carry
# arbitrary values there even if every other field happens to match. Matching
# is done in Python (see below), not in the SQL WHERE clause, so these
# JSON-typed columns compare by value regardless of key order or
# serialization differences across database backends.
DEFAULT_SEED_MATCH_COLUMNS: tuple[str, ...] = (
    "name",
    "description",
    "transport",
    "provider_name",
    "category",
    "icon",
    "is_visible_in_connector",
    "oauth_scopes",
    "launch_config",
)


def delete_unmodified_seeded_rows(
    bind: Connection,
    table: TableClause,
    seed_rows: Iterable[dict[str, Any]],
    match_columns: Sequence[str] = DEFAULT_SEED_MATCH_COLUMNS,
    id_column: str = "app_id",
) -> None:
    """Delete rows a migration seeded, but only the ones that still match its
    seed snapshot exactly (id plus ``match_columns``).

    Columns absent from the current schema (e.g. dropped by a later
    migration) are skipped rather than treated as a mismatch. A row whose id
    matches but whose other fields differ was created or edited by an
    operator after the fact and is left in place. A ``table`` that declares
    only a subset of columns (a common lightweight ``sa.table()`` pattern),
    or a seed row missing one of the match keys, is handled the same way:
    the column/key is skipped rather than raising ``KeyError``.

    If ``match_columns`` is non-empty but none of its columns exist in the
    current schema, provenance can no longer be verified at all, so nothing
    is deleted (matching on ``id_column`` alone would silently reproduce the
    exact bug this helper exists to prevent). Pass ``match_columns=()``
    explicitly to opt into matching on ``id_column`` only.

    Each candidate row is fetched and compared in Python rather than matched
    via the SQL ``WHERE`` clause, so JSON-typed columns compare structurally
    (immune to key-order or whitespace differences between the stored value
    and the seed) instead of relying on backend-specific text/JSON equality.
    """
    inspector = sa.inspect(bind)
    if table.name not in set(inspector.get_table_names()):
        return

    existing_columns = {column["name"] for column in inspector.get_columns(table.name)}
    if id_column not in existing_columns:
        return
    columns_to_match = [
        column for column in match_columns if column in existing_columns
    ]
    if match_columns and not columns_to_match:
        return

    def _column(name: str) -> Any:
        return table.c[name] if name in table.c else sa.column(name)

    id_col = _column(id_column)

    for row in seed_rows:
        if id_column not in row:
            continue

        if columns_to_match:
            candidate = (
                bind.execute(
                    sa.select(*(_column(column) for column in columns_to_match)).where(
                        id_col == row[id_column]
                    )
                )
                .mappings()
                .first()
            )
            if candidate is None:
                continue
            if any(
                column in row and candidate[column] != row[column]
                for column in columns_to_match
            ):
                continue

        bind.execute(sa.delete(table).where(id_col == row[id_column]))
