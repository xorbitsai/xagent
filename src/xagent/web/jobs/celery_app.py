from __future__ import annotations

from importlib import import_module
from typing import Any

from celery import Celery

from ...config import (
    get_background_job_sweep_interval_seconds,
    get_background_job_visibility_timeout_seconds,
    get_celery_broker_url,
    get_celery_result_backend,
)


def create_celery_app() -> Any:
    broker_url = get_celery_broker_url()
    if not broker_url:
        # Celery still needs an app object for imports/tests. Actual enqueue is
        # guarded by XAGENT_CELERY_ENABLED and Compose sets an explicit broker.
        broker_url = "memory://"

    result_backend = get_celery_result_backend()
    visibility_timeout = get_background_job_visibility_timeout_seconds()
    sweep_interval = get_background_job_sweep_interval_seconds()
    app = Celery("xagent", broker=broker_url, backend=result_backend)
    app.conf.update(
        broker_connection_retry_on_startup=True,
        broker_transport_options={"visibility_timeout": visibility_timeout},
        result_backend_transport_options={"visibility_timeout": visibility_timeout},
        task_acks_late=True,
        task_ignore_result=result_backend is None,
        task_reject_on_worker_lost=True,
        task_routes={
            "xagent.web.jobs.tasks.execute_background_job": {
                "queue": "default",
            },
            "xagent.web.jobs.trigger_tasks.scan_due_triggers": {
                "queue": "triggers",
            },
        },
        beat_schedule={
            "scan-due-triggers-and-stale-jobs": {
                "task": "xagent.web.jobs.trigger_tasks.scan_due_triggers",
                "schedule": float(sweep_interval),
            },
        },
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        worker_prefetch_multiplier=1,
    )
    return app


celery_app = create_celery_app()


def register_celery_tasks() -> None:
    """Import task modules so fresh worker imports register every task."""
    for module_name in (
        "xagent.web.jobs.tasks",
        "xagent.web.jobs.trigger_tasks",
    ):
        import_module(module_name)


register_celery_tasks()
