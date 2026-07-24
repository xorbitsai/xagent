"""SDK workforce endpoints: ``POST /v1/workforces/{id}/runs``.

The heaviest external channel for Workforce (#949 / #805). A
workforce-bound ``xag_*`` API key creates runs here; the subsequent
polling / steps / multi-turn append flow reuses the existing
``/v1/chat/tasks/{task_id}`` family via the run's 1:1
``WorkforceRun.task_id`` binding (a workforce-bound key resolves those
task endpoints through its workforce, see ``_resolve_task_or_404`` in
``tasks.py``).

Run creation is delegated to ``services.workforce_runs.create_workforce_run``
with ``source="sdk"`` -- the same service the internal web channel uses,
so worker-delegation semantics, the archive/config turn guard, and
idempotency all behave identically across channels. External metering
counts only run creation (worker delegations are excluded, consistent
with agent-as-tool behavior; LLM cost is captured by the quota system).
"""

import logging
from typing import NoReturn, Tuple

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload, selectinload

from ...models.agent_api_key import AgentApiKey
from ...models.database import get_db
from ...models.user import User
from ...models.workforce import Workforce, WorkforceAgent
from ...schemas.v1 import (
    CreateWorkforceRunRequest,
    CreateWorkforceRunResponse,
)
from ...services.workforce_errors import WorkforceRunErrorCode
from ...services.workforce_runs import create_workforce_run
from .deps import get_workforce_from_api_key, record_key_usage
from .errors import V1ApiError, V1ErrorCode

router = APIRouter(prefix="/workforces")
logger = logging.getLogger(__name__)


def _load_workforce_for_run(db: Session, workforce_id: int) -> Workforce | None:
    """Eager-load the manager agent + workers the run service needs.

    The auth dependency already loaded the ``Workforce`` row, but only
    the bare row -- ``create_workforce_run`` walks ``manager_agent`` (for
    the execution-mode fallback + connector runtime) and ``workers`` (for
    the config snapshot), so re-load with those relationships pinned to
    avoid lazy round-trips mid-run-creation.
    """
    return (
        db.query(Workforce)
        .options(
            joinedload(Workforce.manager_agent),
            selectinload(Workforce.workers).joinedload(WorkforceAgent.agent),
        )
        .filter(Workforce.id == workforce_id)
        .first()
    )


# Stable service-layer code -> (v1 wire code, HTTP status). Switching on
# WorkforceRunError.code -- a machine discriminant -- rather than the
# free-text detail keeps this mapping robust against message rewording,
# mirroring how ``raise_for_turn_rejection`` keys off TaskTurnError.reason.
_SERVICE_CODE_TO_V1: dict[str, tuple[V1ErrorCode, int]] = {
    WorkforceRunErrorCode.ARCHIVED: (V1ErrorCode.WORKFORCE_ARCHIVED, 409),
    WorkforceRunErrorCode.NOT_ACTIVE: (V1ErrorCode.WORKFORCE_NOT_ACTIVE, 409),
    WorkforceRunErrorCode.IDEMPOTENCY_CONFLICT: (
        V1ErrorCode.IDEMPOTENCY_CONFLICT,
        409,
    ),
    WorkforceRunErrorCode.FILE_NOT_FOUND: (V1ErrorCode.FILE_NOT_FOUND, 404),
    WorkforceRunErrorCode.INVALID_EXECUTION_MODE: (V1ErrorCode.INVALID_INPUT, 422),
    WorkforceRunErrorCode.INVALID_IDEMPOTENCY_KEY: (V1ErrorCode.INVALID_INPUT, 422),
}


def _raise_v1_for_workforce_http_error(exc: HTTPException) -> NoReturn:
    """Translate the run service's exception into the v1 envelope.

    ``WorkforceRunError`` carries a stable ``code`` we switch on directly;
    plain ``HTTPException``s (e.g. ``ensure_workforce_access``'s 404/403)
    have no code, so they fall through to a conservative status-based
    mapping.
    """
    detail = str(exc.detail or "")
    code = getattr(exc, "code", None)
    mapped = _SERVICE_CODE_TO_V1.get(code) if code is not None else None
    if mapped is not None:
        v1_code, http_status = mapped
        message = detail or None if http_status == 422 else None
        raise V1ApiError(v1_code, http_status, message=message) from exc

    # Codeless HTTPException fallback. ``ensure_workforce_access`` raises
    # 404 (missing) / 403 (policy) -- both surface as workforce_not_found
    # so "exists but forbidden" isn't observable; other statuses map
    # conservatively.
    status = exc.status_code
    if status in (403, 404):
        raise V1ApiError(V1ErrorCode.WORKFORCE_NOT_FOUND, 404) from exc
    if status == 409:
        raise V1ApiError(V1ErrorCode.TASK_BUSY, 409) from exc
    if status == 400:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT, 422, message=detail or None
        ) from exc
    raise V1ApiError(V1ErrorCode.INTERNAL_ERROR, 500) from exc


@router.post(
    "/{workforce_id}/runs",
    status_code=202,
    response_model=CreateWorkforceRunResponse,
)
async def create_workforce_run_endpoint(
    workforce_id: int,
    request: CreateWorkforceRunRequest,
    authed: Tuple[Workforce, AgentApiKey] = Depends(get_workforce_from_api_key),
    db: Session = Depends(get_db),
) -> CreateWorkforceRunResponse:
    """Create a workforce run and kick off its first turn.

    The ``workforce_id`` path parameter must match the workforce the
    presented key is bound to; a mismatch is a 404 ``workforce_not_found``
    (never 403), so the existence of other workforces isn't observable.

    Returns 202 with the new ``workforce_run_id`` + bound ``task_id``.
    Clients then drive the conversation through the
    ``/v1/chat/tasks/{task_id}`` endpoints (poll status, fetch steps,
    append messages). An ``idempotency_key`` replay returns the original
    run with ``created=false``.

    Raises:
        V1ApiError 401: missing / invalid / revoked / agent-bound key.
        V1ApiError 404: ``workforce_id`` doesn't match the bound workforce.
        V1ApiError 409: workforce archived / not active / idempotency
            conflict / task busy (not retryable except task_busy).
        V1ApiError 422: invalid execution mode / idempotency key.
    """
    bound_workforce, key_row = authed

    # Path/key consistency check. The key already binds a workforce;
    # ``workforce_id`` in the path is required by the REST shape but the
    # bound workforce is the only authority. Mismatch is a 404 -- never a
    # 403 -- so unrelated workforce ids aren't observable to this caller.
    if workforce_id != int(bound_workforce.id):
        raise V1ApiError(V1ErrorCode.WORKFORCE_NOT_FOUND, 404)

    workforce = _load_workforce_for_run(db, workforce_id)
    if workforce is None:
        raise V1ApiError(V1ErrorCode.WORKFORCE_NOT_FOUND, 404)

    # ``create_workforce_run`` acts as the workforce owner. Resolve the
    # user from the workforce's owner_user_id (the key carries no user
    # session of its own).
    owner = db.get(User, int(workforce.owner_user_id))
    if owner is None:
        raise V1ApiError(V1ErrorCode.INTERNAL_ERROR, 500)

    try:
        result = await create_workforce_run(
            db,
            owner,
            workforce,
            message=request.message.content,
            selected_file_ids=request.message.files,
            execution_mode=request.execution_mode,
            # SDK/REST runs stay out of the Web UI task-discovery surface
            # (mirrors POST /v1/chat/tasks), but remain in the workforce's
            # own run history (which queries WorkforceRun, not Task.is_visible).
            is_preview=False,
            is_visible=False,
            source="sdk",
            idempotency_key=request.idempotency_key,
        )
    except HTTPException as exc:
        _raise_v1_for_workforce_http_error(exc)

    task = result.task
    workforce_run = result.workforce_run

    # Usage metering: count run creation as one external call, but only
    # when a run was actually created -- an idempotency replay is a no-op
    # that returns the cached result and must not double-bill. Worker
    # delegations inside the run are never metered here (agent-as-tool
    # parity); LLM cost is tracked by the quota system.
    if result.created:
        record_key_usage(str(key_row.key_prefix))

    return CreateWorkforceRunResponse(
        workforce_run_id=int(workforce_run.id),
        workforce_id=int(workforce.id),
        task_id=int(task.id),
        agent_id=int(task.agent_id),
        status=str(workforce_run.status),
        created=result.created,
        created_at=task.created_at,
        run_id=task.run_id,
        state_version=int(task.state_version or 0),
        control_state=str(task.control_state or "idle"),
    )
