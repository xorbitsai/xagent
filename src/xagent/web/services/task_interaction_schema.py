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
call. A cheaper no-cache option also exists: inspecting the session's own
connection instead of the engine skips a pool checkout and roughly halves
the cost.
"""

from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from ..models.task_interaction import TaskInteractionRequest


def interaction_requests_table_exists(db: Session) -> bool:
    """True when this session's database has task_interaction_requests."""
    return inspect(db.get_bind()).has_table(TaskInteractionRequest.__tablename__)
