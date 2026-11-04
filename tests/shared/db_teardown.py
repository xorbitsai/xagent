"""One drop-the-schema teardown for fixtures that build a real database.

``tasks.last_checkpoint_trace_event_id -> trace_events.id`` and
``trace_events.task_id -> tasks.id`` form a constraint cycle. SQLAlchemy
breaks the cycle for DROP ordering by dropping the ``use_alter=True``
constraint first, which works on PostgreSQL but not on SQLite: that dialect
cannot ALTER a constraint, so ``create_all`` renders the FK inline in
``CREATE TABLE`` and it is still enforced when ``drop_all`` reaches the
tables. SQLite's ``DROP TABLE`` runs an implicit delete of the table's rows,
so with ``foreign_keys=ON`` (this repo's default -- see
``apply_sqlite_concurrency_pragmas``) dropping either side of a *populated*
cycle fails with "FOREIGN KEY constraint failed".

No drop order satisfies a populated cycle, so enforcement is turned off for
the single connection that does the dropping, not for the fixture's tests.
The rows are being discarded, not checked.

Any teardown that drops ``Base.metadata`` against a ``get_engine()`` SQLite
database in a test that persists a checkpoint (a row with
``last_checkpoint_trace_event_id`` set) must use this helper instead of
calling ``Base.metadata.drop_all`` directly.
"""

from __future__ import annotations

from sqlalchemy.engine import Engine

from xagent.web.models.database import Base


def drop_all_tables(engine: Engine) -> None:
    """Drop every ``Base.metadata`` table on ``engine``."""

    if engine.dialect.name != "sqlite":
        Base.metadata.drop_all(bind=engine)
        return

    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        Base.metadata.drop_all(bind=connection)
