from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from ...core.computer.desktop_relay import get_desktop_relay_registry
from ...core.computer.relay import (
    ComputerTargetReadiness,
    build_computer_target_readiness,
    get_browser_relay_registry,
)
from ..auth_dependencies import get_current_user
from ..models.user import User

computer_router = APIRouter(tags=["computer"])


class ComputerReadinessTargets(BaseModel):
    extension_relay: ComputerTargetReadiness
    desktop_relay: ComputerTargetReadiness


class ComputerReadinessResponse(BaseModel):
    targets: ComputerReadinessTargets


@computer_router.get(
    "/api/computer/readiness",
    response_model=ComputerReadinessResponse,
)
async def get_computer_readiness(
    response: Response,
    user: User = Depends(get_current_user),
) -> ComputerReadinessResponse:
    """Return both user-controlled computer targets in one live snapshot."""

    user_id = int(user.id)
    browser_status, desktop_status = await asyncio.gather(
        get_browser_relay_registry().status(user_id),
        get_desktop_relay_registry().status(user_id),
    )
    response.headers["Cache-Control"] = "no-store"
    return ComputerReadinessResponse(
        targets=ComputerReadinessTargets(
            extension_relay=build_computer_target_readiness(
                browser_status,
                target_kind="browser",
            ),
            desktop_relay=build_computer_target_readiness(
                desktop_status,
                target_kind="desktop",
            ),
        )
    )
