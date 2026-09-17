"""Schema-presence predicate for the interaction table.

Deletion and retention paths run on deployments upgraded to a revision
before task_interaction_requests exists. Absence is a normal deployment
state, not a fault: the migration that creates the table has not been
applied yet.

The check is deliberately uncached. Measured on PostgreSQL it costs less
than the trace INSERT plus commit the same checkpoint write already
performs, and it fires once per checkpoint write, not per trace event.
Caching False would be poisonous: the table's migration can be applied
while processes are running, and a process that latched False before the
migration would skip interaction rows in purge (reintroducing the
IntegrityError this module exists to prevent) and stop protecting active
resume anchors in prune, permanently, with no operator remedy short of a
restart. If measurements ever justify caching, the only safe shape is a
one-way latch: cache True alone -- the table is only ever added, never
dropped, so a stale True cannot occur -- and let False re-query on every
call.

The check inspects the session's own connection, not the engine.
Inspecting the engine checks out two connections per call: on SQLite's
NullPool each is a new physical connection that re-runs the WAL,
busy_timeout and foreign_keys connect pragmas (measured ~12x the cost),
and on PostgreSQL each is a pooled checkout taken while the caller
already holds one (~4.5x). Reusing the session's connection also keeps
the gate's answer on the same snapshot as the query it guards. Under
READ COMMITTED -- the isolation level this deployment runs -- the two
forms are indistinguishable, including across a migration applied while
the process is running; under REPEATABLE READ or SERIALIZABLE the
session's snapshot can miss a table created after its transaction began,
where the constraint layer still fails loudly rather than silently.
"""

from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from ..models.task_interaction import TaskInteractionRequest


def interaction_requests_table_exists(db: Session) -> bool:
    """True when this session's database has task_interaction_requests."""
    return inspect(db.connection()).has_table(TaskInteractionRequest.__tablename__)
