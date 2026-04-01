"""change_task_fk_to_cascade

Revision ID: 44a6d3a54c35
Revises: a0f42ff986b2
Create Date: 2026-03-11 00:47:06.197244

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "44a6d3a54c35"
down_revision: Union[str, None] = "a0f42ff986b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _build_uploaded_files_create_sql(
    task_fk_on_delete: str, include_tasks_fk: bool
) -> str:
    """Build SQL to create uploaded_files_new table.

    Args:
        task_fk_on_delete: ON DELETE action for task_id FK (e.g. "CASCADE", "SET NULL")
        include_tasks_fk: Whether to include the FK to tasks table
    """
    task_fk_line = (
        f",\n                FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE {task_fk_on_delete}"
        if include_tasks_fk
        else ""
    )
    return f"""
        CREATE TABLE uploaded_files_new (
            id INTEGER PRIMARY KEY,
            file_id VARCHAR(36) UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            task_id INTEGER,
            filename VARCHAR(512) NOT NULL,
            storage_path VARCHAR(2048) NOT NULL UNIQUE,
            mime_type VARCHAR(255),
            file_size INTEGER NOT NULL DEFAULT 0,
            created_at DATETIME,
            updated_at DATETIME,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE{task_fk_line}
        )
    """


def _recreate_uploaded_files_sqlite(
    existing_tables: list[str], task_fk_on_delete: str
) -> None:
    """Recreate uploaded_files table for SQLite with new FK behavior."""
    include_tasks_fk = "tasks" in existing_tables
    op.execute("PRAGMA foreign_keys=off")

    op.execute(_build_uploaded_files_create_sql(task_fk_on_delete, include_tasks_fk))

    # Copy data from old table to new table
    op.execute("""
        INSERT INTO uploaded_files_new
        SELECT * FROM uploaded_files
    """)

    # Drop old table
    op.execute("DROP TABLE uploaded_files")

    # Rename new table
    op.execute("ALTER TABLE uploaded_files_new RENAME TO uploaded_files")

    # Recreate indexes
    op.execute("CREATE INDEX ix_uploaded_files_id ON uploaded_files (id)")
    op.execute("CREATE INDEX ix_uploaded_files_file_id ON uploaded_files (file_id)")

    op.execute("PRAGMA foreign_keys=on")


def upgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    dialect = bind.dialect.name

    # Check if uploaded_files table exists
    existing_tables = inspector.get_table_names()
    if "uploaded_files" not in existing_tables:
        return

    if dialect == "sqlite":
        _recreate_uploaded_files_sqlite(existing_tables, "CASCADE")
    else:
        # For PostgreSQL and others that support ALTER TABLE
        # Only proceed if tasks table exists
        if "tasks" not in existing_tables:
            return

        fks = inspector.get_foreign_keys("uploaded_files")
        fk_name = None
        for fk in fks:
            if (
                "task_id" in fk["constrained_columns"]
                and fk["referred_table"] == "tasks"
            ):
                fk_name = fk["name"]
                break

        if fk_name:
            op.drop_constraint(fk_name, "uploaded_files", type_="foreignkey")

        op.create_foreign_key(
            "fk_uploaded_files_task_id_tasks",
            "uploaded_files",
            "tasks",
            ["task_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)
    dialect = bind.dialect.name

    # Check if uploaded_files table exists
    existing_tables = inspector.get_table_names()
    if "uploaded_files" not in existing_tables:
        return

    if dialect == "sqlite":
        _recreate_uploaded_files_sqlite(existing_tables, "SET NULL")
    else:
        # For PostgreSQL and others
        # Only proceed if tasks table exists
        if "tasks" not in existing_tables:
            return

        fks = inspector.get_foreign_keys("uploaded_files")
        fk_name = None
        for fk in fks:
            if (
                "task_id" in fk["constrained_columns"]
                and fk["referred_table"] == "tasks"
            ):
                fk_name = fk["name"]
                break

        if fk_name:
            op.drop_constraint(fk_name, "uploaded_files", type_="foreignkey")

        op.create_foreign_key(
            "fk_uploaded_files_task_id_tasks",
            "uploaded_files",
            "tasks",
            ["task_id"],
            ["id"],
            ondelete="SET NULL",
        )
