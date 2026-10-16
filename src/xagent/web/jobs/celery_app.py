from __future__ import annotations

import contextlib
import logging
import re
import subprocess
import sys
from importlib import import_module
from typing import Any

from celery import Celery
from celery.signals import beat_init, worker_init

from ...config import (
    get_background_job_sweep_interval_seconds,
    get_background_job_visibility_timeout_seconds,
    get_celery_broker_url,
    get_celery_result_backend,
    get_kb_milvus_cleanup_interval_seconds,
    validate_encryption_config,
)

logger = logging.getLogger(__name__)


def create_celery_app() -> Any:
    broker_url = get_celery_broker_url()
    if not broker_url:
        # Celery still needs an app object for imports/tests. Actual enqueue is
        # guarded by XAGENT_CELERY_ENABLED and Compose sets an explicit broker.
        broker_url = "memory://"

    result_backend = get_celery_result_backend()
    visibility_timeout = get_background_job_visibility_timeout_seconds()
    sweep_interval = get_background_job_sweep_interval_seconds()
    cleanup_interval = get_kb_milvus_cleanup_interval_seconds()
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
            "xagent.web.jobs.trigger_tasks.purge_leftover_milvus_rows": {
                "queue": "kb",
            },
        },
        beat_schedule={
            "scan-due-triggers-and-stale-jobs": {
                "task": "xagent.web.jobs.trigger_tasks.scan_due_triggers",
                "schedule": float(sweep_interval),
            },
            "purge-leftover-milvus-rows": {
                "task": "xagent.web.jobs.trigger_tasks.purge_leftover_milvus_rows",
                "schedule": float(cleanup_interval),
                # An unconsumed enqueue expires instead of piling up in Redis.
                "options": {"expires": float(cleanup_interval)},
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


@worker_init.connect
@beat_init.connect
def validate_worker_encryption(**_: Any) -> None:
    from ..models.database import create_database_engine
    from ..services.encryption_key_rotation import validate_stored_encryption_keys

    engine = None
    try:
        validate_encryption_config()
        engine = create_database_engine()
        validate_stored_encryption_keys(engine)
    except Exception as exc:
        # Celery swallows ordinary signal exceptions. SystemExit must abort
        # admission; never print raw SQL/parameters or secret values.
        detail = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        raise SystemExit(f"Refusing Celery startup: {detail}") from None
    finally:
        # No inherited DBAPI connections in prefork pool children.
        if engine is not None:
            engine.dispose()


_REFUSED = 3
_LOCK_KB_ENGINE = f"""
import logging
import sys

from xagent.core.tools.core.RAG_tools.storage.vector_backend import (
    lock_deployment_kb_engine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s\\t%(message)s",
    stream=sys.stderr,
    force=True,
)
try:
    lock_deployment_kb_engine()
except Exception as exc:
    print(exc)
    raise SystemExit({_REFUSED})
"""
_LOG_LINE = re.compile(r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)\t(.*)$")


def _parse(line: str, default: int) -> tuple[int, str]:
    match = _LOG_LINE.match(line)
    return (getattr(logging, match[1]), match[2]) if match else (default, line)


def _relay(output: str, default: int) -> None:
    for line in output.splitlines():
        level, text = _parse(line, default)
        with contextlib.suppress(Exception):
            logger.log(level, "KB engine check: %s", text)


@worker_init.connect
def lock_kb_engine_at_worker_start(**_: Any) -> None:
    # A prefork parent that opened LanceDB has pool children that segfault in it.
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _LOCK_KB_ENGINE],
            capture_output=True,
            text=True,
            errors="replace",
        )
    except Exception as exc:
        detail = f"cannot run the engine check: {exc}"
    else:
        _relay(proc.stdout, logging.INFO)
        _relay(proc.stderr, logging.WARNING)
        code = proc.returncode
        if code == 0:
            return
        stdout = proc.stdout.strip().splitlines()
        stderr = proc.stderr.strip().splitlines()
        if code == _REFUSED and stdout:
            detail = stdout[-1]
        else:
            how = (
                f"was killed by signal {-code}"
                if code < 0
                else f"exited with code {code}"
            )
            tail = f": {_parse(stderr[-1], logging.WARNING)[1]}" if stderr else ""
            detail = f"the engine check {how}{tail}"
    # Celery logs and swallows an Exception from a signal handler.
    raise SystemExit(f"Refusing to start the Celery worker: {detail}")


def register_celery_tasks() -> None:
    """Import task modules so fresh worker imports register every task."""
    for module_name in (
        "xagent.web.jobs.tasks",
        "xagent.web.jobs.trigger_tasks",
    ):
        import_module(module_name)


register_celery_tasks()
