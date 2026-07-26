"""SDK management endpoints for built-in templates."""

from typing import Any

from fastapi import APIRouter, Depends, Request

from ...schemas.v1 import V1TemplateDetail, V1TemplateSummary
from .deps import (
    PersonalApiKeySnapshot,
    UserPrincipalSnapshot,
    get_user_from_personal_key,
)
from .errors import V1ApiError, V1ErrorCode

router = APIRouter(prefix="/templates")


def _get_template_manager(request: Request) -> Any:
    template_manager = getattr(request.app.state, "template_manager", None)
    if template_manager is None:
        raise V1ApiError(
            V1ErrorCode.INTERNAL_ERROR,
            500,
            "Template manager is not configured.",
        )
    return template_manager


def _localized(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("en") or next(iter(value.values()), None)
    return value


def _template_summary(template: dict[str, Any]) -> V1TemplateSummary:
    return V1TemplateSummary(
        id=template["id"],
        name=template["name"],
        category=template.get("category", ""),
        featured=bool(template.get("featured", False)),
        description=_localized(template.get("descriptions")) or "",
        features=_localized(template.get("features")) or [],
        connections=template.get("connections") or [],
        setup_time=_localized(template.get("setup_time")) or "5 min setup",
        tags=_localized(template.get("tags")) or [],
        author=template.get("author", ""),
        version=template.get("version", ""),
    )


@router.get("", response_model=list[V1TemplateSummary])
async def list_templates(
    request: Request,
    _authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> list[V1TemplateSummary]:
    template_manager = _get_template_manager(request)
    templates = await template_manager.list_templates()
    return [_template_summary(template) for template in templates]


@router.get("/{template_id}", response_model=V1TemplateDetail)
async def get_template(
    template_id: str,
    request: Request,
    _authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1TemplateDetail:
    template_manager = _get_template_manager(request)
    template = await template_manager.get_template(template_id)
    if template is None:
        raise V1ApiError(V1ErrorCode.TEMPLATE_NOT_FOUND, 404)
    summary = _template_summary(template)
    return V1TemplateDetail(
        **summary.model_dump(),
        agent_config=template.get("agent_config") or {},
    )
