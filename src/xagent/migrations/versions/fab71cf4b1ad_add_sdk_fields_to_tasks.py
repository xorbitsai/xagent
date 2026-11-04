"""add_sdk_fields_to_tasks

Adds four columns to ``tasks`` needed by the public SDK surface:

  - ``input``           TEXT  -- latest-turn user input as plaintext
  - ``output``          TEXT  -- latest-turn assistant output as plaintext
  - ``error_message``   TEXT  -- last failure reason, when status=FAILED
  - ``source``          VARCHAR(20) DEFAULT 'internal'
                              -- call origin classifier: internal / sdk / widget

``source`` defaults to ``'internal'`` so existing rows ALTERed in place
and any future INSERT that doesn't specify it (every legacy code path
in chat.py / websocket.py / widget.py / etc) is automatically classified
as ``'internal'``. The migration also explicitly backfills NULL rows
with ``'internal'`` to keep PostgreSQL and SQLite behavior identical
(SQLite ADD COLUMN with DEFAULT does not always populate existing rows
the same way Postgres does).

``output`` and ``error_message`` are populated by the SDK code paths
(commit D's helper); legacy paths continue to leave them NULL, which is
acceptable because legacy consumers don't read them.

Revision ID: fab71cf4b1ad
Revises: a9dc5acb2a6b
Create Date: 2026-05-12 21:30:26.473199

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "fab71cf4b1ad"
down_revision: Union[str, None] = "a9dc5acb2a6b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    # ``tasks`` is created by SQLAlchemy's ``Base.metadata.create_all()``
    # in production, not by a migration in this repo. The migration test
    # CLI (tests/migrations/test_migration_integration.py) runs migrations
    # against an empty database WITHOUT first calling create_all(), so
    # ``tasks`` may not exist here. Matches the same guard used by
    # 20260509_add_delegate_agent_ids_to_tasks.py and other
    # tasks-touching migrations -- they all no-op when the table is
    # absent so the from-scratch migration test stays green.
    if "tasks" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("tasks")}

    # ADD COLUMN steps are individually guarded so a re-run after a
    # partial failure doesn't blow up. Matches the idempotent-guard
    # pattern in commit 0cc9bdf's agent_api_keys migration.
    if "input" not in existing_columns:
        op.add_column("tasks", sa.Column("input", sa.Text(), nullable=True))
    if "output" not in existing_columns:
        op.add_column("tasks", sa.Column("output", sa.Text(), nullable=True))
    if "error_message" not in existing_columns:
        op.add_column("tasks", sa.Column("error_message", sa.Text(), nullable=True))
    if "source" not in existing_columns:
        op.add_column(
            "tasks",
            sa.Column(
                "source",
                sa.String(length=20),
                server_default="internal",
                nullable=True,
            ),
        )

    # Backfill explicit value for legacy rows. PostgreSQL 11+ stores
    # the DEFAULT in metadata so existing rows already read as
    # 'internal'; SQLite's behavior is less guaranteed depending on the
    # version. An explicit UPDATE makes the data on disk match the
    # advertised default in both engines.
    op.execute("UPDATE tasks SET source = 'internal' WHERE source IS NULL")

    # Index supports queries like:
    #   SELECT count(*) FROM tasks WHERE source = 'sdk' AND created_at > ...
    # used by upcoming SDK adoption metrics. Non-unique; the index name
    # follows the project's existing ``ix_<table>_<column>`` convention.
    existing_indexes = {idx["name"] for idx in inspector.get_indexes("tasks")}
    if "ix_tasks_source" not in existing_indexes:
        op.create_index("ix_tasks_source", "tasks", ["source"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    # Same no-op guard as upgrade(): tasks may not exist when downgrade
    # runs after an upgrade against an empty database in the migration
    # test suite.
    if "tasks" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("tasks")}
    existing_indexes = {idx["name"] for idx in inspector.get_indexes("tasks")}

    if "ix_tasks_source" in existing_indexes:
        op.drop_index("ix_tasks_source", table_name="tasks")
    if "source" in existing_columns:
        op.drop_column("tasks", "source")
    if "error_message" in existing_columns:
        op.drop_column("tasks", "error_message")
    if "output" in existing_columns:
        op.drop_column("tasks", "output")
    if "input" in existing_columns:
        op.drop_column("tasks", "input")
