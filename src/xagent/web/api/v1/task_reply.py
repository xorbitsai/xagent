"""SDK reply authorization and protocol mapping for runner-owned recovery."""

from __future__ import annotations

from datetime import datetime, timezone

from ....core.agent.checkpoint import (
    CheckpointAccessRefusedError,
    CheckpointCorruptError,
    CheckpointReadError,
)
from ...models.database import get_session_local
from ...models.task import TaskStatus
from ...schemas.v1 import ReplyRequest, ReplyResponse
from ...services import task_resume as task_resume_service
from ...services.db_runtime import (
    run_db_io_cancellation_safe,
)
from ...services.llm_utils import AutoModelUnavailableError
from .deps import ApiKeyPrincipal, record_key_usage
from .errors import V1ApiError, V1ErrorCode
from .tasks import _resolve_task_or_404, _validate_owner_scope


def _prepare_reply_context_sync(
    *,
    task_id: int,
    principal: ApiKeyPrincipal,
    request: ReplyRequest,
) -> task_resume_service.TaskReplyInput:
    """Authorize the call and validate the reply body.

    A pure read -- no mutation happens here. The actual state transition
    is the prelease claim in the runner, which owns the recovery lifecycle.
    """
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = _resolve_task_or_404(task_id, principal, db)
        _validate_owner_scope(
            principal,
            request_agent_id=request.agent_id,
            request_workforce_id=request.workforce_id,
        )
        if request.message.files:
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT,
                422,
                message=(
                    "files are not accepted on reply; attach files via a "
                    "follow-up call to the task append endpoint "
                    "(POST .../messages) instead"
                ),
            )
        return task_resume_service.TaskReplyInput(
            task_id=int(task.id),
            agent_id=int(task.agent_id),
            task_owner_user_id=int(task.user_id),
            run_id=str(task.run_id) if task.run_id is not None else None,
            status=task.status,
            text=request.message.content,
        )


async def reply_to_task(
    *,
    task_id: int,
    request: ReplyRequest,
    principal: ApiKeyPrincipal,
) -> ReplyResponse:
    """Resume a ``waiting_for_user`` task with the user's answer.

    Args:
        task_id: Path parameter; the target task's primary key.
        request: Validated :class:`ReplyRequest`.
        principal: Key-bound owner from the auth dependency.

    Returns:
        :class:`ReplyResponse` with ``status='running'`` and the same
        ``run_id`` the task was waiting on.

    Raises:
        V1ApiError 401: missing / invalid / revoked key.
        V1ApiError 404: task not found, not owned by the key, or
            body.agent_id / body.workforce_id doesn't match the bound
            owner.
        V1ApiError 422: body.message.files is non-empty, or
            body.agent_id is missing for an agent-bound key.
        V1ApiError 409: ``task_busy`` (task is RUNNING, or the resume
            lease was lost to a concurrent reply -- retryable);
            ``no_pending_interaction`` (task is not currently waiting
            on a question); ``interaction_not_resumable`` (the task's
            saved progress cannot be resumed -- NOT retryable, the task
            stays in waiting_for_user and the caller must start a new
            task).
        V1ApiError 503: ``temporarily_unavailable`` -- the saved
            progress could not be read due to a transient failure;
            retryable.
        500: any other unexpected error (V1 envelope via the global
            handler), including a lease lost mid-resume.
    """
    ctx = await run_db_io_cancellation_safe(
        lambda: _prepare_reply_context_sync(
            task_id=task_id,
            principal=principal,
            request=request,
        )
    )

    try:
        result = await task_resume_service.resume_task_reply(ctx)
    except task_resume_service.TaskResumeBusyError as exc:
        raise V1ApiError(V1ErrorCode.TASK_BUSY, 409) from exc
    except task_resume_service.TaskResumeNotWaitingError as exc:
        raise V1ApiError(V1ErrorCode.NO_PENDING_INTERACTION, 409) from exc
    except task_resume_service.TaskResumeNotResumableError as exc:
        raise V1ApiError(V1ErrorCode.INTERACTION_NOT_RESUMABLE, 409) from exc
    except CheckpointReadError as exc:
        if isinstance(exc, CheckpointCorruptError):
            raise V1ApiError(V1ErrorCode.INTERACTION_NOT_RESUMABLE, 409) from exc
        if isinstance(exc, CheckpointAccessRefusedError):
            if exc.reason == "superseded_legacy":
                raise V1ApiError(V1ErrorCode.INTERACTION_NOT_RESUMABLE, 409) from exc
            raise V1ApiError(V1ErrorCode.TASK_BUSY, 409) from exc
        # CheckpointUnavailableError, or any future CheckpointReadError
        # subclass this dispatch does not yet know about: treat it
        # conservatively as retryable rather than assuming a terminal
        # failure, so an unrecognized failure mode never silently
        # collapses into a data-losing branch.
        raise V1ApiError(V1ErrorCode.TEMPORARILY_UNAVAILABLE, 503) from exc
    except AutoModelUnavailableError as exc:
        raise V1ApiError(V1ErrorCode.AUTO_MODEL_UNAVAILABLE, 409) from exc

    await record_key_usage(str(principal.key.key_prefix))

    return ReplyResponse(
        task_id=ctx.task_id,
        agent_id=ctx.agent_id,
        workforce_id=(
            int(principal.workforce.id) if principal.workforce is not None else None
        ),
        status=TaskStatus.RUNNING.value,
        accepted_at=datetime.now(timezone.utc),
        run_id=result.run_id,
        state_version=result.state_version,
        control_state=result.control_state,
    )
