"""``Idempotency-Key`` wrapper shared by the ``/v1`` create and append routes."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, NoReturn

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...services import api_idempotency
from ...services.db_runtime import run_db_io_cancellation_safe
from .deps import ApiKeyPrincipal
from .errors import V1ApiError, V1ErrorCode

logger = logging.getLogger(__name__)

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
IDEMPOTENT_REPLAYED_HEADER = "Idempotent-Replayed"
_RETRY_AFTER_SECONDS = "2"


def idempotency_scope(principal: ApiKeyPrincipal) -> tuple[str, int]:
    """The tenant a key is deduplicated within: the bound agent or workforce of a
    runtime key, or the holder of a personal key."""
    if principal.agent is not None:
        return "agent", int(principal.agent.id)
    if principal.workforce is not None:
        return "workforce", int(principal.workforce.id)
    if principal.user is not None:
        return "user", int(principal.user.id)
    raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)


def raise_idempotency_rejection(exc: api_idempotency.IdempotencyRejected) -> NoReturn:
    if exc.reason == "reused":
        raise V1ApiError(V1ErrorCode.IDEMPOTENCY_KEY_REUSED, 422)
    if exc.reason == "in_progress":
        raise V1ApiError(
            V1ErrorCode.IDEMPOTENCY_KEY_IN_PROGRESS,
            409,
            headers={"Retry-After": _RETRY_AFTER_SECONDS},
        )
    if exc.reason == "outcome_unknown":
        raise V1ApiError(
            V1ErrorCode.IDEMPOTENCY_CONFLICT,
            409,
            message=(
                "The original request with this Idempotency-Key created task "
                f"{exc.task_id}, but its outcome was not recorded. Read the task "
                f"with GET /v1/chat/tasks/{exc.task_id} to see its state; a new "
                "task was not started."
            ),
            details={"task_id": exc.task_id},
        )
    raise V1ApiError(V1ErrorCode.IDEMPOTENCY_CONFLICT, 409)


async def run_with_idempotency(
    *,
    principal: ApiKeyPrincipal,
    operation: str,
    header_value: str | None,
    request_payload: dict,
    run: Callable[[api_idempotency.IdempotencyClaim | None], Awaitable[BaseModel]],
) -> BaseModel | JSONResponse:
    """Run ``run`` once per ``Idempotency-Key``.

    ``run`` receives the reservation (``None`` without a header) and must pass
    it to the call that creates the task so the reservation is bound in that
    transaction. Without the header this is exactly ``await run(None)``.
    """
    try:
        key = api_idempotency.validate_idempotency_key(header_value)
    except ValueError as exc:
        raise V1ApiError(V1ErrorCode.IDEMPOTENCY_KEY_INVALID, 422) from exc
    if key is None:
        return await run(None)

    scope_kind, scope_id = idempotency_scope(principal)
    fingerprint = api_idempotency.request_fingerprint(request_payload)
    acquired: list[api_idempotency.IdempotencyClaim] = []

    def acquire() -> (
        api_idempotency.IdempotencyClaim | api_idempotency.IdempotencyReplay
    ):
        outcome = api_idempotency.acquire_sync(
            scope_kind=scope_kind,
            scope_id=scope_id,
            operation=operation,
            key=key,
            fingerprint=fingerprint,
        )
        if isinstance(outcome, api_idempotency.IdempotencyClaim):
            # Recorded inside the worker: the claim is committed even when the
            # awaiting request is cancelled while the worker finishes.
            acquired.append(outcome)
        return outcome

    try:
        outcome = await run_db_io_cancellation_safe(acquire)
    except api_idempotency.IdempotencyRejected as exc:
        raise_idempotency_rejection(exc)
    except BaseException:
        if acquired:
            await _release_quietly(acquired[0])
        raise
    if isinstance(outcome, api_idempotency.IdempotencyReplay):
        return JSONResponse(
            status_code=outcome.status,
            content=outcome.body,
            headers={IDEMPOTENT_REPLAYED_HEADER: "true"},
        )

    claim = outcome
    try:
        response = await run(claim)
    except api_idempotency.IdempotencyRejected as exc:
        # Fenced out: a retry took the reservation over while this request was
        # still running, and nothing was created for it.
        raise_idempotency_rejection(exc)
    except BaseException:
        await _release_quietly(claim)
        raise
    try:
        await run_db_io_cancellation_safe(
            lambda: api_idempotency.complete_sync(
                claim, status=202, body=response.model_dump(mode="json")
            )
        )
    except Exception:
        # The task is already running; failing the request would only invite a
        # retry. The bound reservation keeps that retry from starting another.
        logger.exception("failed to record the Idempotency-Key response")
    return response


async def _release_quietly(claim: api_idempotency.IdempotencyClaim) -> None:
    """Release ``claim`` without letting a storage error replace the exception
    that is already unwinding."""
    try:
        await run_db_io_cancellation_safe(lambda: api_idempotency.release_sync(claim))
    except Exception:
        logger.exception("failed to release the Idempotency-Key reservation")
