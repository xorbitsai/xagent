"""add uploaded-file detach provenance

Revision ID: 20260830_uploaded_file_detach
Revises: 20260828_terminal_cmd_events
Create Date: 2026-08-30
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.interfaces import ReflectedForeignKeyConstraint

revision: str = "20260830_uploaded_file_detach"
down_revision: Union[str, None] = "20260828_terminal_cmd_events"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "uploaded_files"
TASK_FK = "fk_uploaded_files_task_id_tasks"
INDEX = "ix_uploaded_files_detached_gc"
SQLITE_NAMING = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
}
SQLITE_UNIQUE_TABLE_ARGS = (
    sa.UniqueConstraint("file_id", name="uq_uploaded_files_file_id"),
    sa.UniqueConstraint("storage_path", name="uq_uploaded_files_storage_path"),
)


def _task_fk(inspector: sa.Inspector) -> ReflectedForeignKeyConstraint | None:
    return next(
        (
            fk
            for fk in inspector.get_foreign_keys(TABLE)
            if fk.get("constrained_columns") == ["task_id"]
        ),
        None,
    )


def _sqlite_upgrade(inspector: sa.Inspector) -> None:
    task_fk = _task_fk(inspector)
    has_tasks = "tasks" in inspector.get_table_names()
    with op.batch_alter_table(
        TABLE,
        recreate="always",
        naming_convention=SQLITE_NAMING,
        table_args=SQLITE_UNIQUE_TABLE_ARGS,
    ) as batch:
        batch.add_column(sa.Column("detached_reason", sa.String(64), nullable=True))
        batch.add_column(
            sa.Column("detached_at", sa.DateTime(timezone=True), nullable=True)
        )
        if task_fk is not None:
            batch.drop_constraint(
                str(task_fk.get("name") or TASK_FK),
                type_="foreignkey",
            )
        if has_tasks:
            batch.create_foreign_key(
                TASK_FK,
                "tasks",
                ["task_id"],
                ["id"],
                ondelete="SET NULL",
            )
    op.create_index(
        INDEX,
        TABLE,
        ["task_id", "storage_status", "detached_at", "id"],
        unique=False,
    )


def _sqlite_downgrade(inspector: sa.Inspector) -> None:
    task_fk = _task_fk(inspector)
    has_tasks = "tasks" in inspector.get_table_names()
    op.drop_index(INDEX, table_name=TABLE)
    with op.batch_alter_table(
        TABLE,
        recreate="always",
        naming_convention=SQLITE_NAMING,
        table_args=SQLITE_UNIQUE_TABLE_ARGS,
    ) as batch:
        if task_fk is not None:
            batch.drop_constraint(
                str(task_fk.get("name") or TASK_FK),
                type_="foreignkey",
            )
        if has_tasks:
            batch.create_foreign_key(
                TASK_FK,
                "tasks",
                ["task_id"],
                ["id"],
                ondelete="CASCADE",
            )
        batch.drop_column("detached_at")
        batch.drop_column("detached_reason")


def _drop_current_task_fk(inspector: sa.Inspector) -> None:
    task_fk = _task_fk(inspector)
    if task_fk is not None and task_fk.get("name"):
        op.drop_constraint(str(task_fk["name"]), TABLE, type_="foreignkey")


def _postgresql_index_state(bind: sa.Connection) -> tuple[bool, bool] | None:
    row = bind.execute(
        sa.text(
            "SELECT indisvalid, indisunique FROM pg_index "
            "WHERE indexrelid = to_regclass(:index_name)"
        ),
        {"index_name": INDEX},
    ).one_or_none()
    return (bool(row[0]), bool(row[1])) if row is not None else None


def _postgresql_upgrade(inspector: sa.Inspector) -> None:
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if "detached_reason" not in columns:
        op.add_column(TABLE, sa.Column("detached_reason", sa.String(64), nullable=True))
    if "detached_at" not in columns:
        op.add_column(
            TABLE, sa.Column("detached_at", sa.DateTime(timezone=True), nullable=True)
        )

    if "tasks" in inspector.get_table_names():
        _drop_current_task_fk(inspector)
        op.execute(
            sa.text(
                f"ALTER TABLE {TABLE} ADD CONSTRAINT {TASK_FK} "
                "FOREIGN KEY (task_id) REFERENCES tasks (id) "
                "ON DELETE SET NULL NOT VALID"
            )
        )
        with op.get_context().autocommit_block():
            op.execute(sa.text(f"ALTER TABLE {TABLE} VALIDATE CONSTRAINT {TASK_FK}"))

    existing_indexes = {index["name"]: index for index in inspector.get_indexes(TABLE)}
    expected_columns = ["task_id", "storage_status", "detached_at", "id"]
    existing = existing_indexes.get(INDEX)
    if existing is not None and (
        existing.get("column_names") != expected_columns
        or _postgresql_index_state(op.get_bind()) != (True, False)
    ):
        with op.get_context().autocommit_block():
            op.execute(sa.text(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX}"))
        existing = None
    if existing is None:
        with op.get_context().autocommit_block():
            op.create_index(
                INDEX,
                TABLE,
                expected_columns,
                unique=False,
                postgresql_concurrently=True,
            )


def _other_upgrade(inspector: sa.Inspector) -> None:
    op.add_column(TABLE, sa.Column("detached_reason", sa.String(64), nullable=True))
    op.add_column(
        TABLE, sa.Column("detached_at", sa.DateTime(timezone=True), nullable=True)
    )
    if "tasks" in inspector.get_table_names():
        _drop_current_task_fk(inspector)
        op.create_foreign_key(
            TASK_FK,
            TABLE,
            "tasks",
            ["task_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index(
        INDEX,
        TABLE,
        ["task_id", "storage_status", "detached_at", "id"],
        unique=False,
    )


def upgrade() -> None:
    context = op.get_context()
    dialect = context.dialect.name
    if context.as_sql:
        raise RuntimeError(
            "offline detach-provenance migration is not supported; "
            "the task foreign-key name must be inspected online"
        )

    inspector = sa.inspect(op.get_bind())
    if dialect == "sqlite":
        _sqlite_upgrade(inspector)
    elif dialect == "postgresql":
        _postgresql_upgrade(inspector)
    else:
        _other_upgrade(inspector)


def downgrade() -> None:
    context = op.get_context()
    dialect = context.dialect.name
    if context.as_sql:
        raise RuntimeError(
            "offline detach-provenance migration is not supported; "
            "the task foreign-key name must be inspected online"
        )
    if dialect == "sqlite":
        _sqlite_downgrade(sa.inspect(op.get_bind()))
        return

    inspector = sa.inspect(op.get_bind())
    op.drop_index(INDEX, table_name=TABLE)
    _drop_current_task_fk(inspector)
    if "tasks" in inspector.get_table_names():
        op.create_foreign_key(
            TASK_FK,
            TABLE,
            "tasks",
            ["task_id"],
            ["id"],
            ondelete="CASCADE",
        )
    op.drop_column(TABLE, "detached_at")
    op.drop_column(TABLE, "detached_reason")
