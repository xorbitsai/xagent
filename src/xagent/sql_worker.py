"""Minimal isolated SQL execution service.

Run with ``python -m xagent.sql_worker``. The service intentionally has no
dependency on the XAgent web application, database models, Redis, or storage.
Every query runs in a disposable child process so a timeout closes its database
socket instead of leaving an unbounded driver thread in the worker.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import secrets
import threading
from contextlib import asynccontextmanager
from multiprocessing.connection import Connection
from typing import Any, AsyncIterator, Mapping

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, SecretStr

from .config import (
    get_sql_worker_max_concurrency,
    get_sql_worker_max_query_bytes,
    get_sql_worker_max_result_bytes,
    get_sql_worker_max_rows,
    get_sql_worker_query_timeout_seconds,
    get_sql_worker_token,
)

logger = logging.getLogger(__name__)

SENSITIVE_PLATFORM_ENVIRONMENT_VARIABLES = (
    "DATABASE_URL",
    "PGPASSWORD",
    "POSTGRES_PASSWORD",
    "XAGENT_REDIS_URL",
    "XAGENT_CELERY_BROKER_URL",
    "XAGENT_CELERY_RESULT_BACKEND",
    "ENCRYPTION_KEY",
    "XAGENT_SECRET_ENCRYPTION_KEY",
    "XAGENT_JWT_SECRET",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "DOCKER_HOST",
)
SENSITIVE_ENVIRONMENT_SUFFIXES = (
    "_API_KEY",
    "_CREDENTIALS",
    "_PASSWORD",
    "_SECRET",
    "_TOKEN",
)


class QueryRequest(BaseModel):
    connection_name: str = Field(min_length=1, max_length=128)
    connection_url: SecretStr
    query: str = Field(min_length=1)


def assert_isolated_environment(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Refuse startup when platform credentials or host-control handles leak in."""

    source = os.environ if environ is None else environ
    present = sorted(
        name
        for name, value in source.items()
        if value
        and name != "XAGENT_SQL_WORKER_TOKEN"
        and (
            name in SENSITIVE_PLATFORM_ENVIRONMENT_VARIABLES
            or name.endswith(SENSITIVE_ENVIRONMENT_SUFFIXES)
        )
    )
    if present:
        raise RuntimeError(
            "SQL worker isolation violation; remove sensitive environment variables: "
            + ", ".join(present)
        )


def validate_worker_configuration() -> None:
    """Validate the isolation boundary regardless of how ASGI is launched."""

    assert_isolated_environment()
    token = get_sql_worker_token()
    if token is None or len(token) < 32:
        raise RuntimeError(
            "XAGENT_SQL_WORKER_TOKEN must contain at least 32 characters"
        )


def _safe_worker_error(exc: BaseException) -> str:
    if isinstance(exc, ValueError):
        message = str(exc)
        safe_prefixes = (
            "Invalid SQLAlchemy connection URL",
            "SQLite URI mode is not supported",
            "Opaque database target configuration is not allowed",
            "Unable to determine the database target",
            "The platform database cannot be used by the SQL tool",
            "SQL query returned more than",
            "SQL query result exceeded",
        )
        if message.startswith(safe_prefixes):
            return message
    return "Database query failed"


def _query_child(
    pipe: Connection,
    *,
    connection_name: str,
    connection_url: str,
    query: str,
    max_rows: int,
    max_result_bytes: int,
) -> None:
    try:
        # Import in the disposable process. This keeps database drivers and
        # their native state out of the long-lived HTTP worker.
        from .core.tools.core.sql_tool import _execute_sql_query_local

        result = _execute_sql_query_local(
            connection_name,
            query,
            connection_url=connection_url,
            max_rows=max_rows,
            max_result_bytes=max_result_bytes,
        )
        pipe.send((True, result))
    except BaseException as exc:  # noqa: BLE001 - process boundary must settle
        pipe.send((False, _safe_worker_error(exc)))
    finally:
        pipe.close()


def execute_in_disposable_process(request: QueryRequest) -> dict[str, Any]:
    query_bytes = request.query.encode("utf-8")
    if len(query_bytes) > get_sql_worker_max_query_bytes():
        raise HTTPException(status_code=413, detail="SQL query exceeded the size limit")

    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_query_child,
        kwargs={
            "pipe": send,
            "connection_name": request.connection_name,
            "connection_url": request.connection_url.get_secret_value(),
            "query": request.query,
            "max_rows": get_sql_worker_max_rows(),
            "max_result_bytes": get_sql_worker_max_result_bytes(),
        },
        name="xagent-sql-query",
        daemon=True,
    )
    process.start()
    send.close()
    timeout = get_sql_worker_query_timeout_seconds()
    try:
        if not receive.poll(timeout):
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            raise HTTPException(
                status_code=504,
                detail=(
                    "SQL query timed out; its completion state is unknown and it "
                    "must not be retried automatically"
                ),
            )
        ok, payload = receive.recv()
    except EOFError as exc:
        raise HTTPException(status_code=502, detail="SQL query process failed") from exc
    finally:
        receive.close()
        if process.is_alive():
            process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    if not ok:
        raise HTTPException(status_code=400, detail=str(payload))
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502, detail="SQL query process returned invalid data"
        )
    return payload


def _authorize(authorization: str | None) -> None:
    configured = get_sql_worker_token()
    if configured is None:
        raise HTTPException(status_code=503, detail="SQL worker is not configured")
    scheme, separator, supplied = (authorization or "").partition(" ")
    if (
        separator != " "
        or scheme.lower() != "bearer"
        or not supplied
        or not secrets.compare_digest(supplied, configured)
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    validate_worker_configuration()
    yield


app = FastAPI(
    title="XAgent isolated SQL worker",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=_lifespan,
)
_query_slots = threading.BoundedSemaphore(get_sql_worker_max_concurrency())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query")
def query(
    request: QueryRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _authorize(authorization)
    if not _query_slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="SQL worker is at capacity")
    try:
        return execute_in_disposable_process(request)
    finally:
        _query_slots.release()


def main() -> None:
    validate_worker_configuration()
    import uvicorn

    port = int(os.getenv("XAGENT_SQL_WORKER_PORT", "8090"))
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False)


if __name__ == "__main__":
    main()
