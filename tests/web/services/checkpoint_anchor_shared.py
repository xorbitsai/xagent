"""Shared construction for the "upgraded, FK-less" SQLite schema shape used
by test_task_deletion_checkpoint_pointer.py's ``sqlite_upgraded_session`` and
test_task_lease_recovery.py's ``sqlite_no_anchor_fk_session``.

The checkpoint-anchor migration adds ``last_checkpoint_trace_event_id`` via
``add_column`` (without ``create_foreign_key``, which is PostgreSQL-only), so
a SQLite database upgraded through the real Alembic history has no DB-level
FK for that column -- unlike a freshly ``create_all``'d one, which carries
the full model metadata including the FK. Alembic's SQLite batch mode cannot
add or drop that FK without a full table rebuild, so reproducing the
upgraded shape needs the same trick a real migration would use: create the
full schema via create_all, then rebuild ``tasks`` from its own DDL with
just the anchor FK's clause stripped -- the standard SQLite
rename/recreate/copy/drop procedure.

Both fixtures need this exact no-FK shape, so the construction lives here
once; each test file keeps its own thin fixture wrapping this engine in a
sessionmaker with its own session lifetime and teardown.
"""

from __future__ import annotations

import re

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from xagent.web.models.database import Base

# Imported for its side effect: registering the tasks table (and its anchor
# constraint) on Base.metadata, so this module works no matter which test
# module imports it first.
from xagent.web.models.task import Task  # noqa: F401

CHECKPOINT_ANCHOR_FK_NAME = "fk_tasks_last_checkpoint_trace_event_id"

_ANCHOR_FK_CLAUSE = re.compile(
    r",\s*CONSTRAINT "
    + re.escape(CHECKPOINT_ANCHOR_FK_NAME)
    + r" FOREIGN KEY\([^)]*\) REFERENCES [^,]*"
)


def reset_checkpoint_anchor_fk_create_rule() -> None:
    """Work around a cross-dialect SQLAlchemy DDL-compiler quirk before any
    Base.metadata.create_all() call that might run against more than one
    dialect in the same process.

    The anchor FK is use_alter=True (required for tasks/trace_events'
    constraint cycle -- see the model). The first create_all() against a
    dialect that supports ALTER (PostgreSQL) permanently caches a "defer
    this to ALTER" decision on the constraint's shared, dialect-agnostic
    _create_rule attribute. A later create_all() against SQLite (which
    never attempts the ALTER, since that dialect doesn't support it for
    constraints) then honors the same cached decision and silently omits
    the constraint from CREATE TABLE entirely -- it is never created
    inline or via ALTER. Only a live process that create_all()s against
    both dialects hits this (a real deployment binds one dialect for its
    whole lifetime and never triggers it). Resetting the rule before each
    create_all() call makes every fixture's result independent of what
    dialect (if any) create_all() targeted earlier in the process.
    """
    for constraint in Base.metadata.tables["tasks"].constraints:
        if getattr(constraint, "name", None) == CHECKPOINT_ANCHOR_FK_NAME:
            # _create_rule is private SQLAlchemy state. If a release renames
            # or drops it, assigning to the old name still succeeds, leaves
            # the cached decision in place, and hollows out every fixture
            # that calls this helper -- they would keep asserting against a
            # schema that silently lost the constraint. Check first so that
            # becomes a loud failure instead.
            assert hasattr(constraint, "_create_rule"), (
                "SQLAlchemy no longer exposes Constraint._create_rule; the "
                "cross-dialect create_all() workaround in this helper needs "
                "to be rewritten against the current attribute"
            )
            constraint._create_rule = None
            return
    raise AssertionError(f"{CHECKPOINT_ANCHOR_FK_NAME} constraint not found on tasks")


def build_upgraded_sqlite_engine(db_path: object) -> Engine:
    """Build a SQLite engine whose ``tasks`` table has no DB-level FK for
    the checkpoint pointer column, matching a database upgraded through the
    real migration rather than freshly create_all'd.

    ``db_path`` is the sqlite file path (e.g. ``tmp_path / "upgraded.db"``);
    callers pick their own filename so fixtures from different test modules
    never share a file.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    reset_checkpoint_anchor_fk_create_rule()
    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        original_sql = conn.execute(
            text(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
            )
        ).scalar_one()
        stripped_sql, count = _ANCHOR_FK_CLAUSE.subn("", original_sql)
        assert count == 1, (
            f"expected exactly one {CHECKPOINT_ANCHOR_FK_NAME} clause in the "
            f"tasks DDL, found {count}"
        )
        conn.execute(text("ALTER TABLE tasks RENAME TO tasks_old"))
        conn.execute(text(stripped_sql))
        conn.execute(text("INSERT INTO tasks SELECT * FROM tasks_old"))
        conn.execute(text("DROP TABLE tasks_old"))

    return engine
