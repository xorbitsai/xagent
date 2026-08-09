"""Schema-presence predicate for the interaction table.

Deletion and retention paths run on deployments upgraded to a revision
before task_interaction_requests exists. Absence is a normal deployment
state, not a fault: the migration that creates the table has not been
applied yet.
"""

from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from ..models.task_interaction import TaskInteractionRequest


def interaction_requests_table_exists(db: Session) -> bool:
    """True when this session's database has task_interaction_requests."""
    return inspect(db.get_bind()).has_table(TaskInteractionRequest.__tablename__)
