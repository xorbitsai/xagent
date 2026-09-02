"""Celery Beat entrypoints for KB index maintenance (#1557).

Thin wrappers: the work lives in ``services.kb_maintenance``, which the
in-process fallback loop shares.
"""

from __future__ import annotations

from typing import Any

from ..services.kb_maintenance import retrain_kb_vector_indexes, sweep_kb_storage
from .celery_app import celery_app


@celery_app.task(name="xagent.web.jobs.kb_maintenance_tasks.compact_kb_storage")
def compact_kb_storage() -> dict[str, Any]:
    return sweep_kb_storage()


@celery_app.task(name="xagent.web.jobs.kb_maintenance_tasks.retrain_kb_vector_indexes")
def retrain_kb_vector_indexes_task() -> dict[str, Any]:
    return retrain_kb_vector_indexes()
