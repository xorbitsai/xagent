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

import logging
from typing import Any, Iterable, Sequence

import sqlalchemy as sa
from sqlalchemy.engine import Connection
from sqlalchemy.sql.expression import TableClause

logger = logging.getLogger(__name__)

# Every public_mcp_apps field a pre-existing row can plausibly differ on from
# the migration's seed snapshot, including the two JSON-typed columns
# (oauth_scopes, launch_config): a row that predates a migration's app_id
# being seeded (or predates the admin API's built-in protections) can carry
# arbitrary values there even if every other field happens to match. Matching
# is done in Python (see below), not in the SQL WHERE clause, so these
# JSON-typed columns compare by value regardless of key order or
# serialization differences across database backends. This relies on the
# caller's ``table`` declaring these two columns with ``sa.JSON`` (as every
# current caller does); see the function docstring below for what happens
# if a future caller's table does not.
#
# This default is specific to public_mcp_apps (every current caller deletes
# from that table). A future caller for a different table must pass its own
# ``match_columns`` explicitly rather than rely on this default - if that
# table happens to also have same-named columns, they would be compared
# against unrelated seed data.
PUBLIC_MCP_APPS_SEED_MATCH_COLUMNS: tuple[str, ...] = (
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

# The static (non-env-dependent) oauth_providers fields every current built-in
# provider seed migration matches on when guarding its provider-row downgrade
# delete. client_id/client_secret/redirect_uri are env-dependent and
# deliberately excluded.
OAUTH_PROVIDER_SEED_MATCH_COLUMNS: tuple[str, ...] = (
    "name",
    "auth_url",
    "token_url",
    "userinfo_url",
    "user_id_path",
    "email_path",
    "default_scopes",
)


def delete_unmodified_seeded_rows(
    bind: Connection,
    table: TableClause,
    seed_rows: Iterable[dict[str, Any]],
    match_columns: Sequence[str] = PUBLIC_MCP_APPS_SEED_MATCH_COLUMNS,
    id_column: str = "app_id",
    accepted_seed_rows: Iterable[dict[str, Any]] = (),
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
    current schema, or a given seed row supplies none of those columns,
    provenance can no longer be verified at all, so nothing is deleted for
    that row (matching on ``id_column`` alone would silently reproduce the
    exact bug this helper exists to prevent). Pass ``match_columns=()``
    explicitly to opt into matching on ``id_column`` only.

    Each candidate row is fetched and its ``match_columns`` compared in
    Python rather than matched via the SQL ``WHERE`` clause, so JSON-typed
    columns compare structurally (immune to key-order or whitespace
    differences between the stored value and the seed) instead of relying
    on backend-specific text/JSON equality. The final ``DELETE`` then
    re-asserts every non-JSON match column in its own ``WHERE`` clause too,
    so the delete does not rely solely on ``id_column`` being unique and the
    window where a concurrent writer could change the row between the
    ``SELECT`` and the ``DELETE`` is narrowed for those columns. JSON-typed
    columns are left out of that re-assertion (SQL equality on them is the
    same backend-fragile comparison being avoided above), so that specific
    narrow race remains: a concurrent write that changes only a JSON column
    between the two statements is not caught.

    ``accepted_seed_rows`` may contain alternate, sanctioned snapshots left by
    later migrations. They are considered only for an id also present in
    ``seed_rows``. This lets a historical seed downgrade accept both the exact
    row it inserted and a known descendant-migration representation without
    weakening comparison for arbitrary operator changes.

    A JSON-typed column only compares structurally when ``table`` declares
    it with ``sa.JSON``. A ``table`` that leaves a JSON column untyped (the
    "lightweight subset of columns" pattern above) gets that column back as
    a raw driver value instead of a deserialized Python object, so it will
    essentially never compare equal to the seed's Python value. That fails
    safe (the row is preserved, never wrongly deleted) rather than raising,
    but it does mean such a row is never cleaned up on downgrade either.
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
        logger.warning(
            "delete_unmodified_seeded_rows: none of match_columns %s exist on "
            "%s; preserving all seed rows for this downgrade since provenance "
            "cannot be verified",
            list(match_columns),
            table.name,
        )
        return

    def _column(name: str) -> Any:
        return table.c[name] if name in table.c else sa.column(name)

    id_col = _column(id_column)
    match_column_refs = [(column, _column(column)) for column in columns_to_match]
    scalar_column_refs = [
        (column, ref)
        for column, ref in match_column_refs
        if not isinstance(ref.type, sa.JSON)
    ]

    accepted_rows_by_id: dict[Any, list[dict[str, Any]]] = {}
    for accepted_row in accepted_seed_rows:
        if id_column not in accepted_row:
            logger.warning(
                "delete_unmodified_seeded_rows: ignoring an accepted seed row for "
                "%s missing id_column %r",
                table.name,
                id_column,
            )
            continue
        accepted_rows_by_id.setdefault(accepted_row[id_column], []).append(accepted_row)

    for row in seed_rows:
        if id_column not in row:
            logger.warning(
                "delete_unmodified_seeded_rows: skipping a seed row for %s "
                "missing id_column %r",
                table.name,
                id_column,
            )
            continue
        row_id = row[id_column]
        if match_column_refs and not any(
            column in row for column, _ in match_column_refs
        ):
            # The row supplies none of the columns we'd need to verify
            # provenance with, so (symmetrically with the schema-side check
            # above) refuse to delete rather than falling back to matching on
            # id_column alone.
            logger.warning(
                "delete_unmodified_seeded_rows: preserving %s row %s=%r; seed "
                "row supplies none of match_columns %s",
                table.name,
                id_column,
                row_id,
                columns_to_match,
            )
            continue

        delete_conditions = [id_col == row_id]

        if match_column_refs:
            candidate = (
                bind.execute(
                    sa.select(*(ref for _, ref in match_column_refs))
                    .select_from(table)
                    .where(id_col == row_id)
                )
                .mappings()
                .first()
            )
            if candidate is None:
                continue
            accepted_rows = [row, *accepted_rows_by_id.get(row_id, ())]
            matching_row = None
            mismatched_columns: set[str] = set()
            for accepted_row in accepted_rows:
                supplied_columns = [
                    column for column, _ in match_column_refs if column in accepted_row
                ]
                if not supplied_columns:
                    logger.warning(
                        "delete_unmodified_seeded_rows: ignoring an accepted seed "
                        "snapshot for %s row %s=%r; it supplies none of "
                        "match_columns %s",
                        table.name,
                        id_column,
                        row_id,
                        columns_to_match,
                    )
                    continue
                current_mismatches = {
                    column
                    for column in supplied_columns
                    if candidate[column] != accepted_row[column]
                }
                if not current_mismatches:
                    matching_row = accepted_row
                    break
                mismatched_columns.update(current_mismatches)

            if matching_row is None:
                logger.warning(
                    "delete_unmodified_seeded_rows: preserving %s row %s=%r; "
                    "current value differs from every accepted seed snapshot on %s "
                    "(likely operator-modified)",
                    table.name,
                    id_column,
                    row_id,
                    sorted(mismatched_columns),
                )
                continue
            delete_conditions.extend(
                ref == matching_row[column]
                for column, ref in scalar_column_refs
                if column in matching_row
            )

        bind.execute(sa.delete(table).where(sa.and_(*delete_conditions)))
