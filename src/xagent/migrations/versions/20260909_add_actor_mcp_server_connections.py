"""add actor-scoped stdio MCP connection storage

Revision ID: 20260909_actor_mcp_connections
Revises: 20260901_seed_zendesk_mcp_app
Create Date: 2026-09-09
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.web.models.generation import RandomUUID

revision: str = "20260909_actor_mcp_connections"
down_revision: Union[str, None] = "20260901_seed_zendesk_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "actor_mcp_server_connections"

_UNIQUE_CONSTRAINT_COLUMNS = {
    (
        "uq_actor_mcp_server_connections_lifecycle_generation",
        ("lifecycle_generation",),
    ),
    (
        "uq_actor_mcp_server_connections_actor_app",
        ("user_id", "resource_owner_key", "app_id"),
    ),
    (
        "uq_actor_mcp_server_connections_actor_catalog_generation",
        ("user_id", "resource_owner_key", "catalog_app_generation"),
    ),
}


def _compact_sql(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").lower().replace('"', ""))


def _column_type_fingerprint(
    dialect: str, name: str, type_: sa.types.TypeEngine
) -> str:
    visit_name = type_.__visit_name__.lower()
    if name in {"id", "user_id"} and visit_name == "integer":
        return "integer"
    if name in {"lifecycle_generation", "catalog_app_generation"}:
        if (
            dialect == "sqlite"
            and visit_name == "char"
            and getattr(type_, "length", None) == 32
        ):
            return "uuid"
        if dialect == "postgresql" and visit_name == "uuid":
            return "uuid"
    if name == "resource_owner_key" and visit_name == "varchar":
        return f"varchar:{getattr(type_, 'length', None)}"
    if name == "app_id" and visit_name == "varchar":
        return f"varchar:{getattr(type_, 'length', None)}"
    if name == "encrypted_env" and visit_name == "json":
        return "json"
    if name in {"created_at", "updated_at"} and visit_name in {
        "datetime",
        "timestamp",
    }:
        # SQLite does not persist timezone metadata in its declared DATETIME type.
        timezone = (
            False if dialect == "sqlite" else bool(getattr(type_, "timezone", False))
        )
        return f"timestamp:timezone={timezone}"
    return f"unsupported:{type_!r}"


def _default_fingerprint(dialect: str, name: str, value: object) -> str:
    compact = _compact_sql(value)
    if name == "id":
        if dialect == "sqlite" and not compact:
            return "identity"
        expected = "nextval('actor_mcp_server_connections_id_seq'::regclass)"
        if dialect == "postgresql" and compact == expected:
            return "identity"
    if name == "lifecycle_generation":
        if dialect == "postgresql" and compact in {
            "gen_random_uuid()",
            "(gen_random_uuid())",
        }:
            return "random-uuid"
        sqlite_random_uuid = _compact_sql(
            "lower(hex(randomblob(4))) || lower(hex(randomblob(2))) || "
            "'4' || substr(lower(hex(randomblob(2))), 2) || "
            "substr('89ab', (random() & 3) + 1, 1) || "
            "substr(lower(hex(randomblob(2))), 2) || lower(hex(randomblob(6)))"
        )
        if dialect == "sqlite" and compact == sqlite_random_uuid:
            return "random-uuid"
    if name in {"created_at", "updated_at"}:
        allowed = {"current_timestamp"} if dialect == "sqlite" else {"now()"}
        if compact in allowed:
            return "current-timestamp"
    if value is None:
        return "none"
    return f"unsupported:{compact}"


def _check_expression_fingerprint(dialect: str, value: object) -> str:
    compact = _compact_sql(value)
    if dialect == "sqlite" and re.fullmatch(
        r"cast\(lifecycle_generationasvarchar(?:\(\d+\))?\)<>''", compact
    ):
        return "generation-nonempty"
    unwrapped = compact.replace("(", "").replace(")", "")
    if dialect == "postgresql" and re.fullmatch(
        r"lifecycle_generation::(?:charactervarying|text)(?:::text)?<>''::text",
        unwrapped,
    ):
        return "generation-nonempty"
    return f"unsupported:{compact}"


def _index_fingerprint(index: Mapping[str, Any]) -> tuple[object, ...]:
    options = dict(index.get("dialect_options") or {})
    where = options.pop("postgresql_where", None)
    include = tuple(index.get("include_columns") or ())
    option_include = tuple(options.pop("postgresql_include", ()) or ())
    nulls_not_distinct = options.pop("postgresql_nulls_not_distinct", False)
    column_sorting = tuple(
        sorted(
            (name, tuple(values))
            for name, values in dict(index.get("column_sorting") or {}).items()
        )
    )
    return (
        index.get("name"),
        tuple(index.get("column_names") or ()),
        tuple(index.get("expressions") or ()),
        bool(index.get("unique")),
        include,
        option_include,
        column_sorting,
        _compact_sql(where),
        bool(nulls_not_distinct),
        tuple(sorted((str(key), str(value)) for key, value in options.items())),
    )


def _unique_fingerprint(
    dialect: str, constraint: Mapping[str, Any]
) -> tuple[object, ...]:
    options = dict(constraint.get("dialect_options") or {})
    include = tuple(options.pop("postgresql_include", ()) or ())
    nulls_not_distinct = bool(options.pop("postgresql_nulls_not_distinct", False))
    return (
        constraint.get("name"),
        tuple(constraint.get("column_names") or ()),
        include if dialect == "postgresql" else (),
        nulls_not_distinct,
        tuple(sorted((str(key), str(value)) for key, value in options.items())),
    )


def _reflected_schema_fingerprint(inspector: sa.Inspector) -> dict[str, object]:
    dialect = inspector.bind.dialect.name
    columns = inspector.get_columns(TABLE)
    column_fingerprint = tuple(
        (
            column["name"],
            _column_type_fingerprint(dialect, column["name"], column["type"]),
            bool(column["nullable"]),
            _default_fingerprint(dialect, column["name"], column.get("default")),
            column.get("computed"),
            column.get("identity"),
        )
        for column in columns
    )
    uniques = {
        _unique_fingerprint(dialect, constraint)
        for constraint in inspector.get_unique_constraints(TABLE)
    }
    reflected_indexes = inspector.get_indexes(TABLE)
    duplicate_constraints = {
        index.get("duplicates_constraint")
        for index in reflected_indexes
        if index.get("duplicates_constraint") is not None
    }
    indexes = {
        _index_fingerprint(index)
        for index in reflected_indexes
        if index.get("duplicates_constraint") is None
    }
    return {
        "columns": column_fingerprint,
        "primary_key": tuple(
            inspector.get_pk_constraint(TABLE).get("constrained_columns") or ()
        ),
        "uniques": uniques,
        "duplicate_constraints": duplicate_constraints,
        "foreign_keys": {
            (
                tuple(foreign_key.get("constrained_columns") or ()),
                foreign_key.get("referred_schema"),
                foreign_key.get("referred_table"),
                tuple(foreign_key.get("referred_columns") or ()),
                tuple(
                    sorted(
                        (str(key), str(value).upper())
                        for key, value in (foreign_key.get("options") or {}).items()
                    )
                ),
            )
            for foreign_key in inspector.get_foreign_keys(TABLE)
        },
        "checks": {
            (
                constraint.get("name"),
                _check_expression_fingerprint(dialect, constraint.get("sqltext")),
            )
            for constraint in inspector.get_check_constraints(TABLE)
        },
        "indexes": indexes,
    }


def _expected_schema_fingerprint(dialect: str) -> dict[str, object]:
    timestamp_type = (
        "timestamp:timezone=False" if dialect == "sqlite" else "timestamp:timezone=True"
    )
    duplicate_constraints: set[str] = set()
    if dialect == "postgresql":
        duplicate_constraints = {name for name, _columns in _UNIQUE_CONSTRAINT_COLUMNS}
    return {
        "columns": (
            ("id", "integer", False, "identity", None, None),
            ("lifecycle_generation", "uuid", False, "random-uuid", None, None),
            ("user_id", "integer", False, "none", None, None),
            ("resource_owner_key", "varchar:512", False, "none", None, None),
            ("app_id", "varchar:100", False, "none", None, None),
            ("catalog_app_generation", "uuid", False, "none", None, None),
            ("encrypted_env", "json", True, "none", None, None),
            ("created_at", timestamp_type, False, "current-timestamp", None, None),
            ("updated_at", timestamp_type, False, "current-timestamp", None, None),
        ),
        "primary_key": ("id",),
        "uniques": {
            (name, columns, (), False, ())
            for name, columns in _UNIQUE_CONSTRAINT_COLUMNS
        },
        "duplicate_constraints": duplicate_constraints,
        "foreign_keys": {
            (("user_id",), None, "users", ("id",), (("ondelete", "CASCADE"),)),
            (
                ("catalog_app_generation",),
                None,
                "public_mcp_apps",
                ("generation",),
                (("ondelete", "CASCADE"),),
            ),
        },
        "checks": {
            (
                "ck_actor_mcp_server_connections_generation_nonempty",
                "generation-nonempty",
            )
        },
        "indexes": {
            (
                "ix_actor_mcp_server_connections_id",
                ("id",),
                (),
                False,
                (),
                (),
                (),
                "",
                False,
                (),
            )
        },
    }


def _adopt_current_metadata_table() -> bool:
    """Adopt only a dialect-normalized, provably isomorphic metadata table."""
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(TABLE):
        return False
    dialect = inspector.bind.dialect.name
    if dialect not in {"sqlite", "postgresql"}:
        raise RuntimeError(f"unsupported schema-adoption dialect: {dialect}")
    if _reflected_schema_fingerprint(inspector) != _expected_schema_fingerprint(
        dialect
    ):
        raise RuntimeError(
            "actor_mcp_server_connections already exists with incompatible schema"
        )
    return True


def upgrade() -> None:
    if _adopt_current_metadata_table():
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "lifecycle_generation",
            sa.Uuid(),
            server_default=RandomUUID(),
            nullable=False,
        ),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("resource_owner_key", sa.String(length=512), nullable=False),
        sa.Column("app_id", sa.String(length=100), nullable=False),
        sa.Column("catalog_app_generation", sa.Uuid(), nullable=False),
        sa.Column("encrypted_env", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "CAST(lifecycle_generation AS VARCHAR) <> ''",
            name="ck_actor_mcp_server_connections_generation_nonempty",
        ),
        sa.ForeignKeyConstraint(
            ["catalog_app_generation"],
            ["public_mcp_apps.generation"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "lifecycle_generation",
            name="uq_actor_mcp_server_connections_lifecycle_generation",
        ),
        sa.UniqueConstraint(
            "user_id",
            "resource_owner_key",
            "app_id",
            name="uq_actor_mcp_server_connections_actor_app",
        ),
        sa.UniqueConstraint(
            "user_id",
            "resource_owner_key",
            "catalog_app_generation",
            name="uq_actor_mcp_server_connections_actor_catalog_generation",
        ),
    )
    op.create_index(op.f("ix_actor_mcp_server_connections_id"), TABLE, ["id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_actor_mcp_server_connections_id"), table_name=TABLE)
    op.drop_table(TABLE)
