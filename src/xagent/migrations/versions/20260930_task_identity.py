"""Reserve retired SQLite task identities before enabling AUTOINCREMENT.

Revision ID: 20260930_task_identity
Revises: 20260929_task_attachment_detachment

Run online with writers stopped and the deployment's upload roots mounted.
PostgreSQL already allocates task IDs from a sequence and is unchanged.
"""

import logging
import os
import re
from pathlib import Path

import sqlalchemy as sa
from alembic import op

# Alembic loads revisions via spec_from_file_location, outside xagent's package.
from xagent.config import get_external_upload_dirs, get_uploads_dir

revision = "20260930_task_identity"
down_revision = "20260929_task_attachment_detachment"
branch_labels = None
depends_on = None

WORKSPACE_NAME = re.compile(r"(?:web_task_|task_)([0-9]+)\Z")
LEGACY_WORKSPACE_NAME = re.compile(r"task_([0-9]+)\Z")
USER_NAME = re.compile(r"user_[0-9]+\Z")
WORKSPACE_SUBDIRS = {"input", "output", "temp"}
MAX_SAFE_ID = (1 << 53) - 1
logger = logging.getLogger(__name__)


def _safe_id(value, source):
    digits = str(value or 0).lstrip("0") or "0"
    if len(digits) > 16 or int(digits) >= MAX_SAFE_ID:
        raise RuntimeError(
            f"Historical task identity {value!r} from {source} is not safe for "
            "JavaScript clients; inspect and reconcile this history before retrying"
        )
    return int(digits)


def _workspace_id(name, source):
    match = WORKSPACE_NAME.fullmatch(name)
    if match is None:
        return 0
    digits = match[1].lstrip("0") or "0"
    return _safe_id(digits, source)


def _path_floor(value, roots):
    # Both separators support metadata from relocated databases. A workspace
    # needs a category directory and a filename after its identity component.
    parts = re.split(r"[/\\]", value or "")
    path = Path(value or "")
    if path.is_absolute():
        resolved = path.resolve(strict=False)
        for root in roots:
            try:
                parts = list(resolved.relative_to(root.resolve(strict=False)).parts)
                break
            except ValueError:
                continue
    candidates = [
        (index, part)
        for index, part in enumerate(parts[:-2])
        if WORKSPACE_NAME.fullmatch(part)
    ]
    structural = [
        (index, part)
        for index, part in candidates
        if parts[index + 1] in WORKSPACE_SUBDIRS
    ]
    legacy = [
        (index, part)
        for index, part in candidates
        if LEGACY_WORKSPACE_NAME.fullmatch(part)
        and (index == 0 or USER_NAME.fullmatch(parts[index - 1]))
    ]
    if structural and legacy and legacy[0][0] < structural[0][0]:
        raise RuntimeError(
            f"Uploaded-file path {value!r} has ambiguous nested task history; "
            "reconcile the path before retrying"
        )
    if structural:
        return _workspace_id(structural[0][1], f"uploaded-file path {value!r}")
    if legacy:
        return _workspace_id(legacy[0][1], f"uploaded-file path {value!r}")
    if candidates:
        raise RuntimeError(
            f"Uploaded-file path {value!r} has an ambiguous task-like directory; reconcile it before retrying"
        )
    return 0


def _directory_floor(configured):
    roots = []
    for root in configured:
        try:
            roots.append(root.resolve(strict=True))
        except FileNotFoundError:
            logger.warning(
                "Upload root does not exist; skipping identity scan: %s", root
            )
    pending = list(configured)
    visited = set()
    floor = 0
    while pending:
        directory = pending.pop()
        try:
            stat = directory.stat()
        except FileNotFoundError:
            continue
        resolved = directory.resolve(strict=True)
        if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
            raise RuntimeError(
                f"Upload directory link {directory} resolves outside configured upload "
                "roots; configure its target as an external upload root and retry"
            )
        with os.scandir(directory) as entries:
            children = list(entries)
        child_dirs = {
            entry.name for entry in children if entry.is_dir(follow_symlinks=False)
        }
        position = directory.parent.resolve(strict=True) / directory.name
        relative_depths = [
            len(position.relative_to(root).parts)
            for root in roots
            if position == root or position.is_relative_to(root)
        ]
        structural = bool(child_dirs & WORKSPACE_SUBDIRS)
        legacy_position = (
            bool(LEGACY_WORKSPACE_NAME.fullmatch(directory.name))
            and bool(child_dirs)
            and (
                1 in relative_depths
                or (2 in relative_depths and USER_NAME.fullmatch(position.parent.name))
            )
        )
        if structural or legacy_position:
            candidate_id = _workspace_id(directory.name, f"directory {directory}")
            floor = max(floor, candidate_id)
            if candidate_id:
                continue
        identity = (stat.st_dev, stat.st_ino)
        if identity in visited:
            continue
        visited.add(identity)
        pending.extend(Path(entry.path) for entry in children if entry.is_dir())
    logger.info(
        "Scanned upload roots %s; historical directory floor is %d", roots, floor
    )
    return floor


def _historical_floor(connection, inspector):
    roots = [get_uploads_dir(), *get_external_upload_dirs()]
    floor = _safe_id(
        connection.exec_driver_sql("SELECT max(id) FROM tasks").scalar(), "tasks.id"
    )
    tables = set(inspector.get_table_names())
    if connection.exec_driver_sql(
        "SELECT 1 FROM sqlite_master WHERE name='sqlite_sequence'"
    ).first():
        floor = max(
            floor,
            _safe_id(
                connection.exec_driver_sql(
                    "SELECT max(seq) FROM sqlite_sequence WHERE name='tasks'"
                ).scalar(),
                "sqlite_sequence.tasks",
            ),
        )
    # These rows intentionally survive deletion, without a task FK.
    for table in ("task_cleanup_obligations", "expired_task_tombstones"):
        if table in tables:
            floor = max(
                floor,
                _safe_id(
                    connection.exec_driver_sql(
                        f"SELECT max(task_id) FROM {table}"
                    ).scalar(),
                    f"{table}.task_id",
                ),
            )
    if "uploaded_files" in tables:
        columns = {column["name"] for column in inspector.get_columns("uploaded_files")}
        if "task_id" in columns:
            floor = max(
                floor,
                _safe_id(
                    connection.exec_driver_sql(
                        "SELECT max(task_id) FROM uploaded_files"
                    ).scalar(),
                    "uploaded_files.task_id",
                ),
            )
        paths = sorted(columns & {"storage_path", "storage_key", "storage_uri"})
        if paths:
            candidate = " OR ".join(f"instr({path}, 'task_') > 0" for path in paths)
            rows = connection.exec_driver_sql(
                f"SELECT {', '.join(paths)} FROM uploaded_files WHERE {candidate}"
            )
            for row in rows:
                floor = max(floor, *(_path_floor(value, roots) for value in row))
    floor = max(floor, _directory_floor(roots))
    logger.info("Historical SQLite task identity floor is %d", floor)
    return floor


def _task_table(connection):
    table = sa.Table(
        "tasks", sa.MetaData(), autoload_with=connection, resolve_fks=False
    )
    # Reflection promotes inline CHECKs to table constraints. These two must
    # stay on their columns: older event migrations drop the columns directly.
    for column in ("conversation_storage_version", "conversation_event_sequence"):
        for constraint in list(table.constraints):
            if (
                isinstance(constraint, sa.CheckConstraint)
                and constraint.name == f"ck_tasks_{column}"
            ):
                table.constraints.remove(constraint)
                table.c[column].constraints.add(constraint)
    # Recover referential actions and compact-DDL uniqueness directly from
    # SQLite's catalog, which is more complete than SQLAlchemy's DDL parser.
    foreign_keys = {}
    for row in connection.exec_driver_sql("PRAGMA foreign_key_list('tasks')"):
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
    for row in connection.exec_driver_sql("PRAGMA index_list('tasks')"):
        if row[2] and row[3] == "u":
            columns = tuple(
                item[2]
                for item in connection.exec_driver_sql(
                    "SELECT * FROM pragma_index_info(?)", (row[1],)
                )
            )
            if columns not in known:
                table.append_constraint(sa.UniqueConstraint(*columns))
                known.add(columns)
    # Recreate explicit indexes from their original SQL, including expression
    # and partial indexes that reflection may omit. Triggers are saved too.
    table.indexes.clear()
    return table


def upgrade():
    context = op.get_context()
    if context.dialect.name == "postgresql":
        return
    if context.dialect.name != "sqlite":
        raise RuntimeError(
            "Task identity migration supports SQLite and PostgreSQL only"
        )
    if context.as_sql:
        raise RuntimeError(
            "Task identity migration requires an online SQLite connection"
        )
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if not inspector.has_table("tasks"):
        return  # Fresh installations stamp head, then use model metadata.
    if connection.exec_driver_sql("PRAGMA foreign_keys").scalar():
        raise RuntimeError("Run task identity migration through _migration_connection")
    # Start a real SQLite write transaction before any DDL, even under the
    # sqlite3 driver's legacy transaction mode. A failure rolls back the entire
    # table replacement and sequence update, and concurrent writers wait.
    connection.exec_driver_sql("UPDATE tasks SET id=id WHERE 0")
    floor = _historical_floor(connection, inspector)
    objects = (
        connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE tbl_name='tasks' "
            "AND type IN ('index', 'trigger') AND sql IS NOT NULL ORDER BY type, name"
        )
        .scalars()
        .all()
    )
    with op.batch_alter_table(
        "tasks",
        recreate="always",
        copy_from=_task_table(connection),
        table_kwargs={"sqlite_autoincrement": True},
    ):
        pass
    for sql in objects:
        connection.exec_driver_sql(sql)
    connection.exec_driver_sql("DELETE FROM sqlite_sequence WHERE name='tasks'")
    connection.exec_driver_sql(
        "INSERT INTO sqlite_sequence(name, seq) VALUES ('tasks', ?)", (floor,)
    )


def downgrade():
    # Older application versions work with AUTOINCREMENT. Removing it would
    # discard the only record of retired IDs after retained files are collected.
    pass
