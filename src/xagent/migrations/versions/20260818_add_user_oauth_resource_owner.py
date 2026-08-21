"""add actor ownership to builtin OAuth credentials

Existing rows remain ordinary credentials because the new owner key is
nullable and has no server default. The previous owner-blind unique constraint
is replaced by two partial unique indexes:

* null owner: ``(user_id, provider, provider_user_id)``
* actor owner: ``(user_id, resource_owner_key, provider, provider_user_id)``

This preserves SQL's existing null behavior for ``provider_user_id``. The
migration performs no backfill and rewrites no credential values.

Deployment ordering is strict for actor activation. The nullable column and
null-owned rows remain readable by old instances, so PostgreSQL workers may be
rolled after the transactional migration; SQLite workers must stop for the
batch table rebuild. Do not create actor-owned rows until every instance runs
owner-aware code because an old owner-blind reader could select one.

Downgrade is permitted only before actor-owned rows exist. Collapsing two actor
namespaces into the old identity can violate uniqueness and would destroy the
security boundary, so downgrade refuses rather than deleting or merging rows.

Revision ID: 20260818_user_oauth_resource_owner
Revises: 20260819_merge_jira_and_linear_heads
Create Date: 2026-08-18
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.db.migration_support import require_owner_aware_unique_index_dialect

revision: str = "20260818_user_oauth_resource_owner"
down_revision: Union[str, None] = "20260819_merge_jira_and_linear_heads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "user_oauth"
OWNER_COLUMN = "resource_owner_key"
OWNER_LENGTH = 512
OLD_CONSTRAINT = "uq_user_provider_account"
ORDINARY_INDEX = "uq_user_oauth_ordinary_account"
ACTOR_INDEX = "uq_user_oauth_actor_account"
ORDINARY_WHERE = sa.text(f"{OWNER_COLUMN} IS NULL")
ACTOR_WHERE = sa.text(f"{OWNER_COLUMN} IS NOT NULL")
OWNER_INDEX_DEFINITIONS = (
    (
        ORDINARY_INDEX,
        ("user_id", "provider", "provider_user_id"),
        True,
        ORDINARY_WHERE,
    ),
    (
        ACTOR_INDEX,
        ("user_id", OWNER_COLUMN, "provider", "provider_user_id"),
        True,
        ACTOR_WHERE,
    ),
)


def _require_partial_unique_index_support() -> str:
    return require_owner_aware_unique_index_dialect(op.get_bind().dialect.name)


def _table_exists() -> bool:
    return sa.inspect(op.get_bind()).has_table(TABLE)


def _column_names() -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(TABLE)}


def _constraint_names() -> set[str]:
    return {
        constraint["name"]
        for constraint in sa.inspect(op.get_bind()).get_unique_constraints(TABLE)
        if constraint.get("name")
    }


def _index_names() -> set[str]:
    return {
        index["name"]
        for index in sa.inspect(op.get_bind()).get_indexes(TABLE)
        if index.get("name")
    }


def _owner_index_names() -> set[str]:
    return {name for name, _columns, _unique, _predicate in OWNER_INDEX_DEFINITIONS}


def _normalize_index_predicate(predicate: object | None) -> str | None:
    if predicate is None:
        return None
    normalized = " ".join(str(predicate).strip().lower().split())
    while normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1].strip()
    return normalized


def _owner_indexes_are_current(dialect: str) -> bool:
    indexes = {
        str(index["name"]): index
        for index in sa.inspect(op.get_bind()).get_indexes(TABLE)
        if index.get("name")
    }
    for name, columns, unique, predicate in OWNER_INDEX_DEFINITIONS:
        index = indexes.get(name)
        if index is None:
            return False
        options = index.get("dialect_options") or {}
        actual_predicate = options.get(f"{dialect}_where")
        if (
            tuple(index.get("column_names") or ()) != columns
            or bool(index.get("unique")) is not unique
            or _normalize_index_predicate(actual_predicate)
            != _normalize_index_predicate(predicate)
        ):
            return False
    return True


def _sqlite_global_owner_relation_names() -> set[str]:
    rows = op.get_bind().execute(
        sa.text(
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table', 'index', 'view') "
            "AND name IN (:ordinary, :actor)"
        ),
        {
            "ordinary": ORDINARY_INDEX,
            "actor": ACTOR_INDEX,
        },
    )
    return {str(name) for name in rows.scalars()}


def _create_owner_indexes() -> None:
    """Create the owner indexes inside the current migration transaction."""
    for name, columns, unique, predicate in OWNER_INDEX_DEFINITIONS:
        kwargs = (
            {"sqlite_where": predicate, "postgresql_where": predicate}
            if predicate is not None
            else {}
        )
        op.create_index(name, TABLE, columns, unique=unique, **kwargs)


def upgrade() -> None:
    dialect = _require_partial_unique_index_support()
    if not _table_exists():
        return

    columns = _column_names()
    constraints = _constraint_names()
    needs_column = OWNER_COLUMN not in columns
    has_old_constraint = OLD_CONSTRAINT in constraints

    if not needs_column and not has_old_constraint:
        if _owner_indexes_are_current(dialect):
            return
        raise RuntimeError("owner-aware UserOAuth schema has incorrect indexes")
    if not needs_column or not has_old_constraint:
        raise RuntimeError("UserOAuth schema is partially owner-aware")

    if dialect == "sqlite":
        collisions = _sqlite_global_owner_relation_names()
        if collisions:
            names = ", ".join(sorted(collisions))
            raise RuntimeError(
                "owner-aware SQLite schema names already exist before migration: "
                f"{names}"
            )
        if needs_column or has_old_constraint:
            # SQLite cannot drop a named UNIQUE constraint directly. Batch mode
            # rebuilds the table once while preserving every credential row.
            with op.batch_alter_table(TABLE) as batch_op:
                if needs_column:
                    batch_op.add_column(
                        sa.Column(OWNER_COLUMN, sa.String(OWNER_LENGTH))
                    )
                if has_old_constraint:
                    batch_op.drop_constraint(OLD_CONSTRAINT, type_="unique")
    elif dialect == "postgresql":
        if needs_column:
            op.add_column(TABLE, sa.Column(OWNER_COLUMN, sa.String(OWNER_LENGTH)))
    else:  # pragma: no cover - rejected before schema inspection above
        raise AssertionError(f"unsupported dialect: {dialect}")

    # PostgreSQL runs this revision in one transaction. Replacement indexes
    # therefore exist before the old constraint is removed, while any failure
    # rolls back the complete schema transition for a clean retry.
    _create_owner_indexes()
    if dialect == "postgresql" and has_old_constraint:
        op.drop_constraint(OLD_CONSTRAINT, TABLE, type_="unique")


def downgrade() -> None:
    _require_partial_unique_index_support()
    if not _table_exists():
        return

    columns = _column_names()
    if OWNER_COLUMN in columns:
        actor_row = (
            op.get_bind()
            .execute(
                sa.text(
                    f"SELECT 1 FROM {TABLE} WHERE {OWNER_COLUMN} IS NOT NULL LIMIT 1"
                )
            )
            .first()
        )
        if actor_row is not None:
            raise RuntimeError(
                "cannot downgrade while actor-owned UserOAuth rows exist"
            )

    existing_indexes = _index_names()
    for index_name in (ACTOR_INDEX, ORDINARY_INDEX):
        if index_name in existing_indexes:
            op.drop_index(index_name, table_name=TABLE)

    has_old_constraint = OLD_CONSTRAINT in _constraint_names()
    dialect = op.get_bind().dialect.name
    if dialect == "sqlite" and (OWNER_COLUMN in columns or not has_old_constraint):
        with op.batch_alter_table(TABLE) as batch_op:
            if not has_old_constraint:
                batch_op.create_unique_constraint(
                    OLD_CONSTRAINT,
                    ["user_id", "provider", "provider_user_id"],
                )
            if OWNER_COLUMN in columns:
                batch_op.drop_column(OWNER_COLUMN)
    else:
        if not has_old_constraint:
            op.create_unique_constraint(
                OLD_CONSTRAINT,
                TABLE,
                ["user_id", "provider", "provider_user_id"],
            )
        if OWNER_COLUMN in columns:
            op.drop_column(TABLE, OWNER_COLUMN)
