from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from ...config import get_native_browser_app_name, get_native_browser_enabled
from ...core.computer.native_browser_readiness import (
    LocalBrowserReadiness,
    LocalBrowserReadinessIssue,
)
from ...core.computer.native_browser_readiness import (
    get_local_browser_readiness as probe_local_browser_readiness,
)
from ..models.user import User
from .auth import get_current_user

computer_router = APIRouter(prefix="/api/computer", tags=["computer"])


@computer_router.get(
    "/local-browser/readiness",
    response_model=LocalBrowserReadiness,
)
async def get_local_browser_readiness_endpoint(
    response: Response,
    user: User = Depends(get_current_user),
) -> LocalBrowserReadiness:
    """Return whether this administrator can control the configured browser."""

    response.headers["Cache-Control"] = "no-store"
    if not get_native_browser_enabled():
        issue = LocalBrowserReadinessIssue(
            code="disabled",
            message="Local browser is disabled on this Xagent host.",
        )
        return LocalBrowserReadiness(
            ready=False,
            connected=False,
            attached=False,
            application="Local browser",
            issues=[issue],
            message=issue.message,
        )
    if not bool(user.is_admin):
        issue = LocalBrowserReadinessIssue(
            code="not_authorized",
            message=(
                "Local browser is restricted to Xagent administrators because "
                "it controls a signed-in browser on the backend host."
            ),
        )
        return LocalBrowserReadiness(
            ready=False,
            connected=False,
            attached=False,
            application="Local browser",
            issues=[issue],
            message=issue.message,
        )
    try:
        get_native_browser_app_name()
    except ValueError as exc:
        issue = LocalBrowserReadinessIssue(
            code="invalid_configuration",
            message=str(exc),
        )
        return LocalBrowserReadiness(
            ready=False,
            connected=False,
            attached=False,
            application="Local browser",
            issues=[issue],
            message=issue.message,
        )
    return await probe_local_browser_readiness()
