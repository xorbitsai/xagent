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

DEFAULT_SEED_MATCH_COLUMNS: tuple[str, ...] = (
    "name",
    "description",
    "transport",
    "provider_name",
    "category",
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
    operator after the fact and is left in place.
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

    for row in seed_rows:
        conditions = [table.c[id_column] == row[id_column]]
        conditions.extend(table.c[column] == row[column] for column in columns_to_match)
        bind.execute(sa.delete(table).where(sa.and_(*conditions)))
