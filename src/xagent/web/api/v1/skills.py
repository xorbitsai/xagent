"""SDK management endpoints for the key holder's personal skills."""

import asyncio
import functools
import logging
from typing import NoReturn, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ....config import get_max_upload_size_bytes
from ...schemas.v1 import (
    V1SkillCreateRequest,
    V1SkillDeleteResponse,
    V1SkillDetail,
    V1SkillSummary,
    V1SkillUpdateRequest,
)
from ...services.skill_management import (
    SkillManagementRuntime,
    SkillSummarySnapshot,
    is_valid_personal_skill_name,
)
from ..skill_hub import (
    _derive_upload_skill_name,
    _normalize_skill_files,
    _safe_zip_extract,
    _upload_limit_exceeded_detail,
)
from ..skill_hub_registry import _MAX_DOWNLOAD_BYTES
from .deps import (
    PersonalApiKeySnapshot,
    UserPrincipalSnapshot,
    get_user_from_personal_key,
)
from .errors import V1ApiError, V1ErrorCode

router = APIRouter(prefix="/skills")

logger = logging.getLogger(__name__)


def _raise_v1_from_skill_http_error(exc: HTTPException) -> NoReturn:
    """Translate an ``HTTPException`` from the shared skill helpers.

    The helpers are shared with the JWT skill routes and raise bare
    ``HTTPException``; SDK clients switch on ``error.code`` instead.
    """
    message = str(exc.detail)
    if exc.status_code in (400, 413):
        raise V1ApiError(V1ErrorCode.INVALID_INPUT, exc.status_code, message) from exc
    if exc.status_code == 404:
        raise V1ApiError(V1ErrorCode.SKILL_NOT_FOUND, 404) from exc
    if exc.status_code == 409:
        raise V1ApiError(V1ErrorCode.SKILL_ALREADY_EXISTS, 409, message) from exc
    logger.error(
        "Skill helper raised an unexpected HTTP status %s: %r",
        exc.status_code,
        exc.detail,
        exc_info=exc,
    )
    raise V1ApiError(V1ErrorCode.INTERNAL_ERROR, 500) from exc


def _summary(snapshot: SkillSummarySnapshot) -> V1SkillSummary:
    return V1SkillSummary(
        name=snapshot.name,
        description=snapshot.description,
        when_to_use=snapshot.when_to_use,
        tags=list(snapshot.tags),
        source=snapshot.source,
    )


def _require_personal_name(name: str) -> None:
    """Only a valid personal skill name can address a personal skill."""
    if not is_valid_personal_skill_name(name):
        raise V1ApiError(V1ErrorCode.SKILL_NOT_FOUND, 404)


@router.get("", response_model=list[V1SkillSummary])
async def list_skills(
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> list[V1SkillSummary]:
    user, _key = authed
    entries = await SkillManagementRuntime().list_entries(user_id=int(user.id))
    return [_summary(entry) for entry in entries]


@router.post("", response_model=V1SkillSummary)
async def create_skill(
    request: V1SkillCreateRequest,
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1SkillSummary:
    """Create a personal skill from SKILL.md text.

    Repeating a request that already succeeded returns 200 without writing
    again. Only a repeat through the same creation method counts: if the name
    already holds the same content but it was uploaded, this returns 409. A 200
    means the name now holds exactly this content, not that this request
    created it.
    """
    user, key = authed
    try:
        created = await SkillManagementRuntime().create_skill(
            user_id=int(user.id),
            key_prefix=key.key_prefix,
            name=request.name,
            skill_md=request.skill_md,
        )
    except HTTPException as exc:
        _raise_v1_from_skill_http_error(exc)
    return _summary(created)


@router.post("/upload", response_model=V1SkillSummary)
async def upload_skill(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1SkillSummary:
    """Create a personal skill from an uploaded ``.zip`` bundle or ``.md`` file.

    An empty ``name`` is treated as absent and the name is derived from the
    upload. Repeating an upload that already succeeded returns 200 without
    writing again. Only a repeat through the same creation method counts: if
    the name already holds the same content but it was created from SKILL.md
    text, this returns 409. A 200 means the name now holds exactly this
    content, not that this request created it.
    """
    user, key = authed
    try:
        limit = min(get_max_upload_size_bytes(), _MAX_DOWNLOAD_BYTES)
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(
                status_code=413,
                detail=_upload_limit_exceeded_detail(limit),
            )
        filename = (file.filename or "").strip()
        lower = filename.lower()
        if lower.endswith(".zip"):
            # Inflating an archive is CPU work over untrusted input.
            files, zip_root = await asyncio.to_thread(
                functools.partial(
                    _safe_zip_extract, data, bad_zip_status=400, max_bytes=limit
                )
            )
        elif lower.endswith(".md"):
            files, zip_root = {"SKILL.md": data}, ""
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported upload — provide a .zip skill bundle or a SKILL.md file.",
            )
        files = _normalize_skill_files(files, max_bytes=limit)
        skill_name = _derive_upload_skill_name(
            filename, zip_root, files["SKILL.md"], override=name
        )
        uploaded = await SkillManagementRuntime().upload_skill(
            user_id=int(user.id),
            key_prefix=key.key_prefix,
            name=skill_name,
            files=files,
        )
    except HTTPException as exc:
        _raise_v1_from_skill_http_error(exc)
    return _summary(uploaded)


@router.get("/{name}", response_model=V1SkillDetail)
async def get_skill(
    name: str,
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1SkillDetail:
    user, _key = authed
    entry = await SkillManagementRuntime().get_entry(user_id=int(user.id), name=name)
    if entry is None:
        raise V1ApiError(V1ErrorCode.SKILL_NOT_FOUND, 404)
    return V1SkillDetail(
        **_summary(entry).model_dump(), skill_md=entry.skill_md, files=list(entry.files)
    )


@router.put("/{name}", response_model=V1SkillSummary)
async def update_skill(
    name: str,
    request: V1SkillUpdateRequest,
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1SkillSummary:
    user, key = authed
    _require_personal_name(name)
    try:
        updated = await SkillManagementRuntime().update_skill(
            user_id=int(user.id),
            key_prefix=key.key_prefix,
            name=name,
            skill_md=request.skill_md,
        )
    except HTTPException as exc:
        _raise_v1_from_skill_http_error(exc)
    return _summary(updated)


@router.delete("/{name}", response_model=V1SkillDeleteResponse)
async def delete_skill(
    name: str,
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1SkillDeleteResponse:
    user, key = authed
    _require_personal_name(name)
    try:
        await SkillManagementRuntime().delete_skill(
            user_id=int(user.id), key_prefix=key.key_prefix, name=name
        )
    except HTTPException as exc:
        _raise_v1_from_skill_http_error(exc)
    return V1SkillDeleteResponse(name=name)
