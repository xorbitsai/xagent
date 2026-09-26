"""The internal web API's answer for a task the retention purge expired (#2565).

Internal routes addressed by task id answer a missing task with ``404`` and a
plain-string ``detail``. For a task retention expired they answer ``410 Gone``
instead, with the structured ``detail`` the rest of this API uses for errors a
client branches on (``{"code": ..., "message": ...}`` plus the fields that
code needs)::

    {
        "code": "task_expired",
        "message": "...",
        "task_id": 42,
        "expired_at": "2026-09-26T11:20:41+00:00",
    }

``410`` keeps an unaware client on the error path it already takes for
``404``, while a client that knows the code can say "expired" instead of "not
found". The v1 API reports the same state through its own error envelope;
this is only the internal shape.

Building the error is the easy half. Whether a caller may *receive* it is the
route's own access predicate applied to the tombstone -- see
``services/expired_tasks.find_expired_task`` -- and a caller who fails it must
get the route's ordinary not-found, or the ``410`` would disclose that the id
existed.
"""

from __future__ import annotations

from fastapi import HTTPException

from ..models.expired_task import ExpiredTaskTombstone
from ..utils.db_timezone import format_datetime_for_api

TASK_EXPIRED_CODE = "task_expired"


def task_expired_http_error(
    tombstone: ExpiredTaskTombstone, *, message: str
) -> HTTPException:
    """``410 task_expired`` for a tombstone the caller is allowed to see."""
    return HTTPException(
        status_code=410,
        detail={
            "code": TASK_EXPIRED_CODE,
            "message": message,
            "task_id": int(tombstone.task_id),
            "expired_at": format_datetime_for_api(tombstone.expired_at),
        },
    )
