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

# Includes every public_mcp_apps field an operator can actually edit on an
# otherwise-builtin row (description/icon/category/is_visible_in_connector;
# see admin_mcp.py's _BUILTIN_PROTECTED_FIELDS), so an admin who customizes
# just one of those fields is still detected as "modified" and preserved.
# oauth_scopes/launch_config are left out: the admin API treats them as
# protected/code-managed for a builtin row (so they can't diverge from the
# seed on their own), and JSON-value equality is comparison-fragile across DB
# backends.
DEFAULT_SEED_MATCH_COLUMNS: tuple[str, ...] = (
    "name",
    "description",
    "transport",
    "provider_name",
    "category",
    "icon",
    "is_visible_in_connector",
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

    for row in seed_rows:
        if id_column not in row:
            continue
        conditions = [_column(id_column) == row[id_column]]
        conditions.extend(
            _column(column) == row[column]
            for column in columns_to_match
            if column in row
        )
        bind.execute(sa.delete(table).where(sa.and_(*conditions)))
