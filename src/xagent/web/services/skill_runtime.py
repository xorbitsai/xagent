"""Request-to-worker database session handoff for Skill runtime reads."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal, NoReturn

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from xagent.skills.library import (
    SkillScopeContext,
    SkillScopeMetadataValue,
    SkillWriteContext,
    SkillWriteProvider,
    SkillWriteProviderError,
    SkillWriteProviderErrorReason,
)
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.database import get_db, release_db_connection_if_clean
from xagent.web.models.user import User

logger = logging.getLogger(__name__)


class SkillRuntimeSessionBoundaryError(RuntimeError):
    """The caller Session cannot yield its connection to a Skill worker."""


def handoff_skill_runtime_session(caller_db: Session) -> None:
    """Release a clean caller transaction before worker-owned database I/O."""
    if not release_db_connection_if_clean(caller_db):
        raise SkillRuntimeSessionBoundaryError(
            "Cannot start Skill runtime database work while the caller "
            "database session has pending writes"
        )


def build_detached_skill_scope(
    *,
    user_id: int | None,
    metadata: Mapping[str, SkillScopeMetadataValue] | None = None,
) -> SkillScopeContext:
    """Build provider identity from explicit detached scalar inputs only."""
    return SkillScopeContext(
        user_id=user_id,
        metadata={} if metadata is None else metadata,
    )


def build_runtime_skill_scope(
    *,
    user_id: int | None,
    metadata: Mapping[str, SkillScopeMetadataValue] | None = None,
    caller_db: Session,
) -> SkillScopeContext:
    """Detach request identity, then hand the caller's pool slot to the worker."""
    context = build_detached_skill_scope(user_id=user_id, metadata=metadata)
    handoff_skill_runtime_session(caller_db)
    return context


def get_skill_runtime_scope(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SkillScopeContext:
    """Resolve detached Skill identity before the route body starts."""
    return build_runtime_skill_scope(
        user_id=int(current_user.id) if current_user.id is not None else None,
        caller_db=db,
    )


async def skill_runtime_session_boundary_error_handler(
    request: Request, exc: SkillRuntimeSessionBoundaryError
) -> JSONResponse:
    """Return one stable public response for an incomplete session handoff."""
    logger.error(
        "Skill runtime session boundary failed for %s",
        request.url.path,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Skill runtime is temporarily unavailable."},
    )


_SKILL_WRITE_FAILURE_STATUS = {
    SkillWriteProviderErrorReason.FORBIDDEN: 403,
    SkillWriteProviderErrorReason.INVALID_REQUEST: 400,
}


def _raise_sanitized_skill_write_provider_error(
    method: str,
    exc: Exception,
    *,
    malformed_public_failure: bool,
) -> NoReturn:
    """Log a provider fault without allowing its data into the HTTP response."""
    if malformed_public_failure:
        logger.error(
            "Skill write provider operation %s returned an invalid public failure",
            method,
        )
    else:
        logger.exception("Skill write provider operation %s failed", method)
    raise HTTPException(
        status_code=500,
        detail="Skill provider operation failed.",
    ) from exc


async def invoke_skill_write_provider(
    provider: SkillWriteProvider | None,
    method: Literal["create_skill", "update_skill_file", "delete_skill"],
    context: SkillWriteContext,
    **kwargs: Any,
) -> Any:
    """Invoke one write operation through its public-error boundary."""
    if provider is None:
        raise HTTPException(
            status_code=400,
            detail="No skill writer is registered for this scope.",
        )
    try:
        operation = getattr(provider, method)
        return await operation(context, **kwargs)
    except SkillWriteProviderError as exc:
        if type(exc) is not SkillWriteProviderError:
            _raise_sanitized_skill_write_provider_error(
                method,
                exc,
                malformed_public_failure=True,
            )
        attributes = vars(exc)
        reason = attributes.get("reason")
        public_detail = attributes.get("public_detail")
        if (
            type(reason) is not SkillWriteProviderErrorReason
            or type(public_detail) is not str
        ):
            _raise_sanitized_skill_write_provider_error(
                method,
                exc,
                malformed_public_failure=True,
            )
        status_code = _SKILL_WRITE_FAILURE_STATUS[reason]
        raise HTTPException(
            status_code=status_code,
            detail=public_detail,
        ) from exc
    except Exception as exc:
        _raise_sanitized_skill_write_provider_error(
            method,
            exc,
            malformed_public_failure=False,
        )
