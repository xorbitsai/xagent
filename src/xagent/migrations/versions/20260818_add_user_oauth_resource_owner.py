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
Revises: 20260818_seed_stripe_mcp_app
Create Date: 2026-08-18
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.db.migration_support import require_owner_aware_unique_index_dialect

revision: str = "20260818_user_oauth_resource_owner"
down_revision: Union[str, None] = "20260818_seed_stripe_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "user_oauth"
SQLITE_BATCH_TEMP_TABLE = "_alembic_tmp_user_oauth"
USERS_TABLE = "users"
USER_CASCADE_FK = "fk_user_oauth_user_id_users"
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
        ORDINARY_WHERE,
    ),
    (
        ACTOR_INDEX,
        ("user_id", OWNER_COLUMN, "provider", "provider_user_id"),
        ACTOR_WHERE,
    ),
)


def _require_partial_unique_index_support() -> str:
    return require_owner_aware_unique_index_dialect(op.get_bind().dialect.name)


def _table_exists() -> bool:
    return sa.inspect(op.get_bind()).has_table(TABLE)


def _users_table_exists() -> bool:
    return sa.inspect(op.get_bind()).has_table(USERS_TABLE)


def _column_names() -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(TABLE)}


def _owner_column_is_current() -> bool:
    owner_column = next(
        (
            column
            for column in sa.inspect(op.get_bind()).get_columns(TABLE)
            if column["name"] == OWNER_COLUMN
        ),
        None,
    )
    if (
        owner_column is None
        or owner_column.get("nullable") is not True
        or owner_column.get("default") is not None
    ):
        return False
    owner_type = owner_column.get("type")
    return isinstance(owner_type, sa.String) and owner_type.length == OWNER_LENGTH


def _constraint_names() -> set[str]:
    return {
        constraint["name"]
        for constraint in sa.inspect(op.get_bind()).get_unique_constraints(TABLE)
        if constraint.get("name")
    }


def _user_cascade_fk_is_current() -> bool:
    """Return whether user deletion is guaranteed to remove every OAuth row."""
    return any(
        tuple(foreign_key.get("constrained_columns") or ()) == ("user_id",)
        and foreign_key.get("referred_table") == USERS_TABLE
        and tuple(foreign_key.get("referred_columns") or ()) == ("id",)
        and str((foreign_key.get("options") or {}).get("ondelete")).upper() == "CASCADE"
        for foreign_key in sa.inspect(op.get_bind()).get_foreign_keys(TABLE)
    )


def _index_names() -> set[str]:
    return {
        index["name"]
        for index in sa.inspect(op.get_bind()).get_indexes(TABLE)
        if index.get("name")
    }


def _normalize_index_predicate(predicate: object | None) -> str | None:
    if predicate is None:
        return None
    normalized = " ".join(str(predicate).strip().lower().split())
    while normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1].strip()
    return normalized


def _missing_owner_index_definitions(
    dialect: str,
) -> tuple[tuple[str, tuple[str, ...], object], ...]:
    """Return missing indexes after rejecting every malformed definition."""
    indexes = {
        str(index["name"]): index
        for index in sa.inspect(op.get_bind()).get_indexes(TABLE)
        if index.get("name")
    }
    missing: list[tuple[str, tuple[str, ...], object]] = []
    for definition in OWNER_INDEX_DEFINITIONS:
        name, columns, predicate = definition
        index = indexes.get(name)
        if index is None:
            missing.append(definition)
            continue
        options = index.get("dialect_options") or {}
        actual_predicate = options.get(f"{dialect}_where")
        if (
            tuple(index.get("column_names") or ()) != columns
            or not bool(index.get("unique"))
            or _normalize_index_predicate(actual_predicate)
            != _normalize_index_predicate(predicate)
        ):
            raise RuntimeError("owner-aware UserOAuth schema has incorrect indexes")
    return tuple(missing)


def _owner_indexes_are_current(dialect: str) -> bool:
    return not _missing_owner_index_definitions(dialect)


def _sqlite_batch_temp_table_name() -> str | None:
    """Return the stored temp-table name with case-insensitive matching.

    SQLite resolves schema identifiers case-insensitively, but ``sqlite_master``
    compares names with binary collation unless the query specifies otherwise.
    """
    name = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' "
                "AND name COLLATE NOCASE = :name LIMIT 1"
            ),
            {"name": SQLITE_BATCH_TEMP_TABLE},
        )
        .scalar_one_or_none()
    )
    return str(name) if name is not None else None


def _reject_sqlite_batch_temp_table(dialect: str) -> None:
    """Reject an interrupted batch rebuild before any upgrade or downgrade work."""
    if dialect != "sqlite":
        return
    table_name = _sqlite_batch_temp_table_name()
    if table_name is None:
        return
    raise RuntimeError(
        "interrupted SQLite UserOAuth rebuild left temporary table "
        f"{table_name}; restore the verified backup or have a database operator "
        "inspect both tables before retrying"
    )


def _sqlite_global_owner_relation_names(
    expected_names: Sequence[str] = (ORDINARY_INDEX, ACTOR_INDEX),
) -> set[str]:
    """Return stored names that collide in SQLite's case-insensitive namespace."""
    rows = op.get_bind().execute(
        sa.text(
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table', 'index', 'view') "
            "AND name COLLATE NOCASE IN (:ordinary, :actor)"
        ),
        {
            "ordinary": ORDINARY_INDEX,
            "actor": ACTOR_INDEX,
        },
    )
    expected = {name.casefold() for name in expected_names}
    actual_names = {str(name) for name in rows.scalars()}
    return {name for name in actual_names if name.casefold() in expected}


def _create_owner_indexes(
    definitions: tuple[tuple[str, tuple[str, ...], object], ...] = (
        OWNER_INDEX_DEFINITIONS
    ),
) -> None:
    """Create selected owner indexes inside the current migration transaction."""
    for name, columns, predicate in definitions:
        kwargs = (
            {"sqlite_where": predicate, "postgresql_where": predicate}
            if predicate is not None
            else {}
        )
        op.create_index(name, TABLE, columns, unique=True, **kwargs)


def upgrade() -> None:
    dialect = _require_partial_unique_index_support()
    _reject_sqlite_batch_temp_table(dialect)
    if not _table_exists():
        return

    if not _users_table_exists():
        raise RuntimeError(
            "owner-aware UserOAuth migration requires the users table; "
            "initialize metadata-owned core tables before running Alembic"
        )

    columns = _column_names()
    constraints = _constraint_names()
    needs_column = OWNER_COLUMN not in columns
    has_old_constraint = OLD_CONSTRAINT in constraints
    needs_user_cascade_fk = not _user_cascade_fk_is_current()

    if not needs_column and not has_old_constraint:
        if not _owner_column_is_current():
            raise RuntimeError(
                "owner-aware UserOAuth schema has incorrect owner column"
            )
        if needs_user_cascade_fk:
            raise RuntimeError(
                "owner-aware UserOAuth schema is missing its user cascade foreign key"
            )
        missing_indexes = _missing_owner_index_definitions(dialect)
        if not missing_indexes:
            return
        if dialect != "sqlite":
            raise RuntimeError("owner-aware UserOAuth schema has incorrect indexes")
        collisions = _sqlite_global_owner_relation_names(
            tuple(name for name, *_rest in missing_indexes)
        )
        if collisions:
            names = ", ".join(sorted(collisions))
            raise RuntimeError(
                "owner-aware SQLite schema names already exist before migration: "
                f"{names}"
            )
        # SQLite batch DDL is not reliably transactional under pysqlite. A
        # process can exit after the old constraint is removed or after only
        # the first replacement index is created. Existing indexes were
        # validated above, so creating only the missing definitions safely
        # completes that exact interrupted state without accepting drift.
        _create_owner_indexes(missing_indexes)
        if not _owner_indexes_are_current(dialect):  # pragma: no cover - invariant
            raise RuntimeError("owner-aware UserOAuth schema has incorrect indexes")
        return
    if not needs_column or not has_old_constraint:
        raise RuntimeError("UserOAuth schema is partially owner-aware")

    # The partial-schema guard above proves that this is the exact legacy
    # shape: the owner column is absent and the old constraint is present.
    if dialect == "sqlite":
        collisions = _sqlite_global_owner_relation_names()
        if collisions:
            names = ", ".join(sorted(collisions))
            raise RuntimeError(
                "owner-aware SQLite schema names already exist before migration: "
                f"{names}"
            )
        # SQLite cannot drop a named UNIQUE constraint directly. Batch mode
        # rebuilds the table once while preserving every credential row.
        with op.batch_alter_table(TABLE) as batch_op:
            batch_op.add_column(sa.Column(OWNER_COLUMN, sa.String(OWNER_LENGTH)))
            if needs_user_cascade_fk:
                batch_op.create_foreign_key(
                    USER_CASCADE_FK,
                    USERS_TABLE,
                    ["user_id"],
                    ["id"],
                    ondelete="CASCADE",
                )
            batch_op.drop_constraint(OLD_CONSTRAINT, type_="unique")
    elif dialect == "postgresql":
        op.add_column(TABLE, sa.Column(OWNER_COLUMN, sa.String(OWNER_LENGTH)))
        if needs_user_cascade_fk:
            op.create_foreign_key(
                USER_CASCADE_FK,
                TABLE,
                USERS_TABLE,
                ["user_id"],
                ["id"],
                ondelete="CASCADE",
            )
    else:  # pragma: no cover - rejected before schema inspection above
        raise AssertionError(f"unsupported dialect: {dialect}")

    # PostgreSQL runs this revision in one transaction. Replacement indexes
    # therefore exist before the old constraint is removed, while any failure
    # rolls back the complete schema transition for a clean retry.
    _create_owner_indexes()
    if dialect == "postgresql":
        op.drop_constraint(OLD_CONSTRAINT, TABLE, type_="unique")


def downgrade() -> None:
    dialect = _require_partial_unique_index_support()
    _reject_sqlite_batch_temp_table(dialect)
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
