"""Record task attachment detachment without classifying historical drafts.

Revision ID: 20260929_task_attachment_detachment
Revises: 20260905_execution_event_writers

Run online, before deploying writers of the new columns. PostgreSQL installs
the FK NOT VALID in a short transaction and releases that DDL lock before
validation. SQLite rebuilds the upload table and requires a maintenance window.
No detached sweep/index is enabled by this foundation revision.
"""

import sqlalchemy as sa
from alembic import op

revision = "20260929_task_attachment_detachment"
down_revision = "20260905_execution_event_writers"
branch_labels = None
depends_on = None

TABLE = "uploaded_files"
FK = "fk_uploaded_files_task_id_tasks"
NAMING = {"fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"}
ORPHAN_SAMPLE_LIMIT = 10


def _inspect():
    context = op.get_context()
    if context.as_sql:
        raise RuntimeError(
            "Attachment detachment migration requires an online connection"
        )
    if context.dialect.name not in {"sqlite", "postgresql"}:
        raise RuntimeError("Attachment detachment supports SQLite and PostgreSQL only")
    return sa.inspect(op.get_bind())


def _task_foreign_key(inspector):
    matches = [
        fk
        for fk in inspector.get_foreign_keys(TABLE)
        if fk["constrained_columns"] == ["task_id"]
    ]
    if not matches:
        return None
    if len(matches) != 1 or matches[0]["referred_table"] != "tasks":
        raise RuntimeError("Expected one uploaded_files.task_id foreign key to tasks")
    return matches[0]


def _raise_for_dangling_task_references(inspector):
    if not inspector.has_table("tasks"):
        return
    rows = inspector.bind.execute(
        sa.text(
            "SELECT uploaded.id, uploaded.task_id "
            "FROM uploaded_files AS uploaded "
            "LEFT JOIN tasks AS task ON task.id = uploaded.task_id "
            "WHERE uploaded.task_id IS NOT NULL AND task.id IS NULL "
            "ORDER BY uploaded.id LIMIT :limit"
        ),
        {"limit": ORPHAN_SAMPLE_LIMIT + 1},
    ).all()
    if not rows:
        return

    sample = rows[:ORPHAN_SAMPLE_LIMIT]
    uploaded_file_ids = [row[0] for row in sample]
    missing_task_ids = list(dict.fromkeys(row[1] for row in sample))
    sample_note = " (sample)" if len(rows) > ORPHAN_SAMPLE_LIMIT else ""
    raise RuntimeError(
        "Attachment detachment migration found uploaded_files.task_id references "
        "to missing tasks. Reconcile these references while preserving the uploaded "
        "file IDs and rows, then rerun the migration. Restore the missing task rows "
        "from an authoritative source or correct each task_id to its verified task; "
        "do not set task_id to NULL or invent detachment markers. "
        f"Affected uploaded file IDs{sample_note}: {uploaded_file_ids}; "
        f"missing task IDs{sample_note}: {missing_task_ids}"
    )


def _sqlite_table():
    """Preserve constraints SQLite reflection can miss on compact legacy DDL."""
    connection = op.get_bind()
    table = sa.Table(
        TABLE,
        sa.MetaData(naming_convention=NAMING),
        autoload_with=connection,
        resolve_fks=False,
    )
    # SQLite's own catalog retains referential actions even when SQLAlchemy's
    # CREATE TABLE parser misses them (for example, after ADD COLUMN).
    foreign_keys = {}
    for row in connection.exec_driver_sql(f"PRAGMA foreign_key_list('{TABLE}')"):
        foreign_keys.setdefault(row[0], []).append(row)
    for rows in foreign_keys.values():
        rows.sort(key=lambda row: row[1])
        columns = tuple(row[3] for row in rows)
        constraint = next(
            fk
            for fk in table.foreign_key_constraints
            if tuple(fk.columns.keys()) == columns
        )
        constraint.onupdate = rows[0][5]
        constraint.ondelete = rows[0][6]
    known = {
        tuple(c.columns.keys())
        for c in table.constraints
        if isinstance(c, sa.UniqueConstraint)
    }
    for row in connection.exec_driver_sql(f"PRAGMA index_list('{TABLE}')"):
        if row[2] and row[3] == "u":
            quoted = connection.dialect.identifier_preparer.quote(row[1])
            columns = tuple(
                item[2]
                for item in connection.exec_driver_sql(f"PRAGMA index_info({quoted})")
            )
            if columns not in known:
                table.append_constraint(sa.UniqueConstraint(*columns))
                known.add(columns)
    return table


def _change_foreign_key(action):
    inspector = _inspect()
    if not inspector.has_table("tasks"):
        # Historical migration-only schemas omit this metadata-owned parent.
        return
    foreign_key = _task_foreign_key(inspector)
    name = (foreign_key["name"] if foreign_key else None) or FK
    current_action = (
        foreign_key.get("options", {}).get("ondelete", "").upper()
        if foreign_key
        else None
    )
    if inspector.bind.dialect.name == "sqlite":
        if current_action == action:
            return
        with op.batch_alter_table(
            TABLE,
            recreate="always",
            naming_convention=NAMING,
            copy_from=_sqlite_table(),
        ) as batch:
            if foreign_key:
                batch.drop_constraint(name, type_="foreignkey")
            batch.create_foreign_key(FK, "tasks", ["task_id"], ["id"], ondelete=action)
        return

    if current_action != action:
        if foreign_key:
            op.drop_constraint(name, TABLE, type_="foreignkey")
        op.create_foreign_key(
            FK,
            TABLE,
            "tasks",
            ["task_id"],
            ["id"],
            ondelete=action,
            postgresql_not_valid=True,
        )
        name = FK
    # Also finish validation after an interrupted prior attempt. Entering this
    # block commits ADD/DROP before the scan takes its weaker validation lock.
    quoted_name = op.get_bind().dialect.identifier_preparer.quote(name)
    with op.get_context().autocommit_block():
        op.execute(sa.text(f"ALTER TABLE {TABLE} VALIDATE CONSTRAINT {quoted_name}"))


def upgrade():
    inspector = _inspect()
    if not inspector.has_table(TABLE):
        return  # Fresh installs use Base.metadata.create_all after stamping.
    _raise_for_dangling_task_references(inspector)
    columns = {c["name"] for c in inspector.get_columns(TABLE)}
    for column in (
        sa.Column("detached_reason", sa.String(32), nullable=True),
        sa.Column("detached_at", sa.DateTime(timezone=True), nullable=True),
    ):
        if column.name not in columns:
            op.add_column(TABLE, column)
    _change_foreign_key("SET NULL")


def downgrade():
    inspector = _inspect()
    if not inspector.has_table(TABLE):
        return
    columns = {c["name"] for c in inspector.get_columns(TABLE)}
    markers = columns & {"detached_reason", "detached_at"}
    if markers:
        # An automatic downgrade must not discard outstanding retention
        # obligations. Export/reconcile them before deliberately rolling back.
        condition = " OR ".join(f"{column} IS NOT NULL" for column in sorted(markers))
        pending = (
            op.get_bind()
            .execute(sa.text(f"SELECT 1 FROM {TABLE} WHERE {condition} LIMIT 1"))
            .first()
        )
        if pending:
            raise RuntimeError("Reconcile detached attachments before downgrading")
    _change_foreign_key("CASCADE")
    with op.batch_alter_table(
        TABLE,
        copy_from=_sqlite_table() if inspector.bind.dialect.name == "sqlite" else None,
    ) as batch:
        for column in ("detached_at", "detached_reason"):
            if column in columns:
                batch.drop_column(column)
