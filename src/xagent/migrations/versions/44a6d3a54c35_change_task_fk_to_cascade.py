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
        # SQLite doesn't support ALTER CONSTRAINT directly
        # Need to recreate the table with the new foreign key
        op.execute("PRAGMA foreign_keys=off")

        # Check if tasks table exists before creating FK
        if "tasks" in existing_tables:
            # Create new table with CASCADE constraint to tasks
            op.execute("""
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
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
                )
            """)
        else:
            # Create new table without tasks FK
            op.execute("""
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
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

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
        # Revert back to SET NULL
        op.execute("PRAGMA foreign_keys=off")

        # Check if tasks table exists before creating FK
        if "tasks" in existing_tables:
            # Create table with SET NULL constraint to tasks
            op.execute("""
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
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE SET NULL
                )
            """)
        else:
            # Create table without tasks FK
            op.execute("""
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
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

        # Copy data
        op.execute("""
            INSERT INTO uploaded_files_new
            SELECT * FROM uploaded_files
        """)

        # Drop and rename
        op.execute("DROP TABLE uploaded_files")
        op.execute("ALTER TABLE uploaded_files_new RENAME TO uploaded_files")

        # Recreate indexes
        op.execute("CREATE INDEX ix_uploaded_files_id ON uploaded_files (id)")
        op.execute("CREATE INDEX ix_uploaded_files_file_id ON uploaded_files (file_id)")

        op.execute("PRAGMA foreign_keys=on")
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
