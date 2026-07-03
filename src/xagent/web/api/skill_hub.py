"""Skill Hub API — manage user-installed skills (saas closed-source).

The Hub composes three capabilities on top of xagent's existing skill
machinery (``SkillManager`` + ``SkillParser``):

  1. **Local skill management** — list / detail / delete the skills
     currently visible to the SkillManager, tagging each with
     ``source`` (builtin / user / external) so the UI can gate
     destructive operations on user-installed skills only.

  2. **ClawHub registry browse & install** — a thin proxy in front of
     ``https://clawhub.ai/api/v1/*`` (the public, anonymous-readable
     OpenClaw skill registry). v0 install policy: skills flagged
     ``"malicious"`` or in moderation state ``"quarantined"``/``"revoked"``
     are refused server-side; never trust the client to honor a
     "are you sure?" prompt for malware.

  3. **In-UI authoring** — write a new SKILL.md from scratch
     (``POST /create``) or edit an installed one in place
     (``PUT /installed/{name}``). Edits and creates both invalidate
     the same cache the chat runtime reads from.

GitHub-URL import was removed in this iteration: we previously
shipped a ``git clone --depth=1`` path, but ClawHub gives us trusted
binaries with provenance and scan results, so we don't need to
re-implement that surface area. If someone really wants an
unscanned-source install path back, ``git`` is still on the box.

All writes (installs, creates, edits) persist to the database via
``UserSkill`` / ``UserSkillFile`` models.  The ``XagentPersonalDbSkillProvider``
(``skills/personal_db.py``) surfaces them back to the SkillManager; because
scoped managers are built fresh per request (no per-user cache), changes are
visible immediately on the next API call without an explicit ``reload()``.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from xagent.web.api.skill_hub_registry import (
    _MAX_DOWNLOAD_BYTES,
    SkillRegistry,
    all_registries,
    get_registry,
)
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.database import get_db
from xagent.web.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skill-hub", tags=["skill-hub"])


# ──────────────────────────────────────────────────────────────────────
# Schemas — local
# ──────────────────────────────────────────────────────────────────────


class SkillSummary(BaseModel):
    """List-view payload for ``GET /installed``."""

    name: str
    description: str = ""
    when_to_use: str = ""
    tags: List[str] = Field(default_factory=list)
    source: str  # "builtin" | "user" | "external"
    scope: Optional[str] = None
    effective: bool = True
    shadowed_by: Optional[str] = None


class SkillDetail(SkillSummary):
    """Detail-view payload for ``GET /installed/{name}``."""

    content: str = ""
    execution_flow: str = ""
    files: List[str] = Field(default_factory=list)
    path: str


_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class CreateSkillRequest(BaseModel):
    """``POST /create`` body. Name is the on-disk directory name; the
    frontmatter ``name`` inside ``skill_md`` is ignored by the parser
    (xagent always uses the dir name as the source of truth)."""

    name: str = Field(..., min_length=1, max_length=64)
    skill_md: str = Field(..., min_length=1, max_length=200_000)
    scope: str = Field("personal", pattern="^(personal|team)$")


class EditSkillRequest(BaseModel):
    """``PUT /installed/{name}`` body. ``name`` is taken from the URL;
    only the SKILL.md content is mutable in v0."""

    skill_md: str = Field(..., min_length=1, max_length=200_000)


# ──────────────────────────────────────────────────────────────────────
# Schemas — registry (ClawHub proxy)
# ──────────────────────────────────────────────────────────────────────


class RegistrySkillSummary(BaseModel):
    """Card-view payload for a ClawHub skill. We forward only the
    fields the UI actually renders so the frontend contract is stable
    even if upstream evolves."""

    slug: str
    displayName: str = ""
    summary: str = ""
    version: Optional[str] = None
    ownerHandle: Optional[str] = None
    installs: Optional[int] = None
    # ClawHub sends this as a unix-ms integer (e.g. 1778485729679),
    # not a string — the frontend formats it. Typed as int.
    updatedAt: Optional[int] = None
    # Trust badge: "clean" / "suspicious" / "malicious" / None
    scanStatus: Optional[str] = None
    # If installed locally already, the local skill name (so UI can
    # show "Installed" instead of an Install button).
    installedAs: Optional[str] = None


class RegistrySkillDetail(BaseModel):
    """Detail payload returned by ``GET /registry/{slug}``."""

    slug: str
    displayName: str = ""
    summary: str = ""
    version: Optional[str] = None
    ownerHandle: Optional[str] = None
    homepage: Optional[str] = None
    readme: Optional[str] = None  # the SKILL.md body if upstream exposes one
    scanStatus: Optional[str] = None
    moderation: Optional[Dict[str, Any]] = None
    installedAs: Optional[str] = None
    registrySource: str = "clawhub"
    # Raw upstream blob for any UI bits we don't have a typed slot for
    # yet (provenance, capability tags, etc.). UI can poke at this for
    # secondary detail panels.
    raw: Dict[str, Any] = Field(default_factory=dict)


class RegistryListResponse(BaseModel):
    items: List[RegistrySkillSummary]
    nextCursor: Optional[str] = None


class InstallSkillRequest(BaseModel):
    slug: str = Field(..., min_length=1, max_length=128)
    version: Optional[str] = None
    scope: str = Field("personal", pattern="^(personal|team)$")


# ──────────────────────────────────────────────────────────────────────
# Helpers — local skill paths
# ──────────────────────────────────────────────────────────────────────


def _user_skills_root() -> Path:
    """The single writable skills directory we install into. Mirrors
    the third root ``skills/utils._get_default_skill_dirs`` configures
    so anything we write here is picked up by the same SkillManager
    every other code path uses."""
    from xagent.core.storage.manager import get_storage_root

    root = get_storage_root() / "skills"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _builtin_skills_root() -> Path:
    from xagent.skills.manager import SkillManager

    return SkillManager.get_builtin_root().resolve()


def _classify_source(skill_path: str) -> str:
    """Tag a skill as ``builtin`` / ``user`` / ``external`` based on
    where on disk it lives."""
    if not skill_path:
        return "external"
    p = Path(skill_path).resolve()
    user = _user_skills_root().resolve()
    builtin = _builtin_skills_root()
    if str(p).startswith(str(builtin) + "/") or p == builtin:
        return "builtin"
    if str(p).startswith(str(user) + "/"):
        return "user"
    return "external"


def _validate_skill_name(name: str) -> None:
    if not name or not _NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Skill name must match [A-Za-z0-9_-]+ (no spaces, slashes, or dots)."
            ),
        )


async def _get_manager(request: Request) -> Any:
    """Return the process-wide SkillManager singleton from app.state.
    This manager holds only filesystem (builtin / external / project) records.
    Per-request scoped views add the personal-DB layer on top; see
    ``_get_scoped_manager``.  Typed as ``Any`` to keep the skills package
    out of this module's import graph."""
    mgr = getattr(request.app.state, "skill_manager", None)
    if mgr is None:
        from xagent.skills.utils import create_skill_manager

        mgr = create_skill_manager()
        request.app.state.skill_manager = mgr
    await mgr.ensure_initialized()
    return mgr


def _scope_context(request: Request, user: User, db: Any) -> Any:
    from xagent.skills.library import SkillScopeContext

    metadata: dict[str, Any] = {}
    team_id = getattr(user, "_saas_team_id", None)
    if isinstance(team_id, int):
        metadata["team_id"] = team_id
    return SkillScopeContext(
        user=user,
        user_id=int(user.id) if user.id is not None else None,
        db=db,
        request=request,
        metadata=metadata,
    )


async def _get_scoped_manager(request: Request, user: User, db: Any) -> Any:
    """Build a per-request SkillManager (no persistent per-user cache).

    Caching strategy — decouple by volatility:

    *Default path* (no custom provider registered):
      - Filesystem records (builtin / external / project skills) are stable, so
        we reuse the records already loaded by the process-wide
        ``app.state.skill_manager`` via ``StaticRecordsProvider``.
      - Personal-DB records are volatile, so ``XagentPersonalDbSkillProvider``
        is queried fresh on every request (cheap: one SQL query).

    *Custom-provider path* (SaaS / overlay installed via
    ``set_skill_library_provider``): the provider is used as-is with the
    user context so that team-scoped records are included.  Each request
    still gets its own ``SkillManager`` instance, so there is no shared
    mutable state between concurrent requests.

    In both paths:
    * No stale-delete bug — the DB layer is always re-queried.
    * No unbounded memory — no persistent per-user dict.
    * No concurrency hazard — each request owns its manager instance.
    """
    from xagent.skills.library import (
        CompositeSkillLibraryProvider,
        StaticRecordsProvider,
        get_skill_library_provider,
    )
    from xagent.skills.manager import SkillManager
    from xagent.skills.personal_db import XagentPersonalDbSkillProvider

    ctx = _scope_context(request, user, db)

    custom_provider = get_skill_library_provider()
    if custom_provider is not None:
        # Custom (e.g. SaaS) provider — use as-is; it handles all layers.
        mgr = SkillManager(provider=custom_provider, context=ctx)
    else:
        # Default path: cached FS records + fresh personal-DB per request.
        global_mgr = await _get_manager(request)
        fs_records = [
            info["_record"]
            for info in global_mgr._skills_cache.values()
            if "_record" in info
        ]
        provider = CompositeSkillLibraryProvider(
            [StaticRecordsProvider(fs_records), XagentPersonalDbSkillProvider()]
        )
        mgr = SkillManager(provider=provider, context=ctx)

    await mgr.reload()
    return mgr


def _skill_to_summary(skill_dict: dict) -> SkillSummary:
    return SkillSummary(
        name=skill_dict["name"],
        description=skill_dict.get("description", ""),
        when_to_use=skill_dict.get("when_to_use", ""),
        tags=skill_dict.get("tags", []),
        source=_summary_source(skill_dict),
        scope=skill_dict.get("scope"),
        effective=bool(skill_dict.get("effective", True)),
        shadowed_by=skill_dict.get("shadowed_by"),
    )


def _skill_to_detail(skill_dict: dict) -> SkillDetail:
    return SkillDetail(
        name=skill_dict["name"],
        description=skill_dict.get("description", ""),
        when_to_use=skill_dict.get("when_to_use", ""),
        tags=skill_dict.get("tags", []),
        source=_summary_source(skill_dict),
        scope=skill_dict.get("scope"),
        effective=bool(skill_dict.get("effective", True)),
        shadowed_by=skill_dict.get("shadowed_by"),
        content=skill_dict.get("content", ""),
        execution_flow=skill_dict.get("execution_flow", ""),
        files=skill_dict.get("files", []),
        path=skill_dict.get("path", ""),
    )


def _summary_source(skill_dict: dict) -> str:
    scope = skill_dict.get("scope")
    if scope == "personal":
        return "user"
    if isinstance(scope, str) and scope:
        return scope
    return skill_dict.get("source") or _classify_source(skill_dict.get("path", ""))


def _normalize_skill_files(files: dict[str, bytes]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    total = 0
    for raw_path, content in files.items():
        path = str(raw_path).replace("\\", "/").lstrip("/")
        if not path or ".." in path.split("/"):
            raise HTTPException(
                status_code=400,
                detail="Skill file path contains a path-traversal sequence.",
            )
        if path.startswith("."):
            raise HTTPException(
                status_code=400, detail="Skill file path must not start with a dot."
            )
        total += len(content)
        if total > _MAX_DOWNLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail="Skill files exceed size budget."
            )
        out[path] = bytes(content)
    if "SKILL.md" not in out:
        raise HTTPException(status_code=400, detail="Skill has no SKILL.md.")
    return out


def _write_personal_skill(
    *,
    db: Any,
    user: User,
    name: str,
    files: dict[str, bytes],
    origin: str = "custom",
    clawhub_slug: str | None = None,
    clawhub_version: str | None = None,
) -> None:
    from xagent.skills.library import guess_media_type
    from xagent.web.models.skill import UserSkill, UserSkillFile

    _validate_skill_name(name)
    user_id = int(user.id)
    normalized = _normalize_skill_files(files)
    existing = (
        db.query(UserSkill)
        .filter(UserSkill.user_id == user_id, UserSkill.name == name)
        .first()
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"A personal skill named {name!r} already exists.",
        )
    skill = UserSkill(
        user_id=user_id,
        name=name,
        origin=origin,
        clawhub_slug=clawhub_slug,
        clawhub_version=clawhub_version,
        created_by_user_id=user_id,
        updated_by_user_id=user_id,
    )
    db.add(skill)
    db.flush()
    for path, content in sorted(normalized.items()):
        db.add(
            UserSkillFile(
                skill_id=skill.id,
                path=path,
                content=content,
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                media_type=guess_media_type(path),
            )
        )
    db.commit()


def _update_personal_skill_md(*, db: Any, user: User, name: str, skill_md: str) -> None:
    from xagent.skills.library import guess_media_type
    from xagent.web.models.skill import UserSkill, UserSkillFile

    skill = (
        db.query(UserSkill)
        .filter(UserSkill.user_id == int(user.id), UserSkill.name == name)
        .first()
    )
    if skill is None:
        raise HTTPException(status_code=404, detail="Personal skill not found")
    content = skill_md.encode("utf-8")
    file = next((item for item in skill.files if item.path == "SKILL.md"), None)
    if file is None:
        file = UserSkillFile(skill_id=skill.id, path="SKILL.md")
        db.add(file)
    file.content = content
    file.size_bytes = len(content)
    file.sha256 = hashlib.sha256(content).hexdigest()
    file.media_type = guess_media_type("SKILL.md")
    skill.updated_by_user_id = int(user.id)
    db.commit()


def _delete_personal_skill(*, db: Any, user: User, name: str) -> None:
    from xagent.web.models.skill import UserSkill

    skill = (
        db.query(UserSkill)
        .filter(UserSkill.user_id == int(user.id), UserSkill.name == name)
        .first()
    )
    if skill is None:
        raise HTTPException(status_code=404, detail="Personal skill not found")
    db.delete(skill)
    db.commit()


def _summary_from_registry_item(
    item: dict, installed_names: set[str], registry: SkillRegistry
) -> RegistrySkillSummary:
    """Normalize one item from ``/api/v1/skills`` or ``/api/v1/search``
    into our typed summary.

    Upstream shape (sampled 2026-05 from clawhub.ai/api/v1/skills):
      {
        slug, displayName, summary,
        tags: {latest: "1.0.0", ...},          ← channel dict, NOT a list!
        stats: {installsCurrent, downloads, stars, ...},
        latestVersion: {version, createdAt, ...},
        metadata: {...},
        createdAt, updatedAt                    ← unix ms
      }

    Search results use the same top-level fields plus ``score`` /
    ``ownerHandle`` (list responses don't carry ownerHandle, only
    detail does). ``scanStatus`` is almost always null today —
    install-time gating happens server-side, not here.
    """
    slug = str(item.get("slug") or "")
    stats = item.get("stats") or {}
    return RegistrySkillSummary(
        slug=slug,
        displayName=str(item.get("displayName") or item.get("name") or slug),
        summary=str(item.get("summary") or item.get("description") or ""),
        version=(
            (item.get("latestVersion") or {}).get("version")
            or (item.get("tags") or {}).get("latest")
            or item.get("version")
        ),
        ownerHandle=item.get("ownerHandle") or (item.get("owner") or {}).get("handle"),
        installs=stats.get("installsCurrent") or item.get("installs"),
        updatedAt=item.get("updatedAt"),
        # ``security`` is almost always missing on list responses
        # today (the registry only attaches it after a scan runs).
        # Read both possible locations defensively.
        scanStatus=registry.extract_scan_status(item),
        installedAs=slug if slug in installed_names else None,
    )


def _installed_slugs(mgr: Any) -> set[str]:
    """Names of skills currently in the SkillManager cache. ClawHub
    slugs and local skill dir names line up because we install to
    ``<user_root>/<slug>/``, so a string-equal check is enough."""
    return set(mgr._skills_cache.keys())  # noqa: SLF001 — internal but stable


def _safe_zip_to_files(zip_bytes: bytes) -> dict[str, bytes]:
    """Read a ClawHub ZIP into a normalized skill file bundle."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=502, detail="ClawHub returned a bad ZIP."
        ) from exc

    total = 0
    raw_files: dict[str, bytes] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        if info.file_size > _MAX_DOWNLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Skill ZIP member too large.")
        total += info.file_size
        if total > _MAX_DOWNLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail="Skill ZIP exceeds size budget."
            )
        path = info.filename.replace("\\", "/").lstrip("/")
        if not path or ".." in path.split("/"):
            raise HTTPException(
                status_code=400, detail="Skill ZIP contains unsafe paths."
            )
        raw_files[path] = zf.read(info)

    skill_md_paths = sorted(
        path for path in raw_files if path.endswith("/SKILL.md") or path == "SKILL.md"
    )
    if not skill_md_paths:
        raise HTTPException(
            status_code=400, detail="ClawHub artifact has no SKILL.md anywhere in it."
        )
    skill_root = skill_md_paths[0].removesuffix("SKILL.md").rstrip("/")
    files: dict[str, bytes] = {}
    for path, content in raw_files.items():
        if skill_root:
            prefix = skill_root + "/"
            if not path.startswith(prefix):
                continue
            rel = path[len(prefix) :]
        else:
            rel = path
        if rel:
            files[rel] = content
    return _normalize_skill_files(files)


def _check_registry_security_gate(registry: Any, detail: dict) -> None:
    """Raise HTTP 403 if the registry flags this skill as unsafe.

    Checks two independent signals:
    * ``scan_status == "malicious"`` — AV/scanner verdict via
      ``registry.extract_scan_status``
    * ``moderation.moderationState in {"quarantined", "revoked"}`` — human
      moderation verdict embedded directly in the detail payload
    """
    scan_status = registry.extract_scan_status(detail)
    moderation = detail.get("moderation") or {}
    moderation_state = (
        moderation.get("moderationState") if isinstance(moderation, dict) else None
    )
    if scan_status == "malicious":
        raise HTTPException(
            status_code=403,
            detail=f"Install refused: this skill is flagged malicious by {registry.display_name} scanners.",
        )
    if moderation_state in ("quarantined", "revoked"):
        raise HTTPException(
            status_code=403,
            detail=f"Install refused: skill is {moderation_state} by {registry.display_name} moderators.",
        )


# ──────────────────────────────────────────────────────────────────────
# Routes — local skills (list / detail / delete)
# ──────────────────────────────────────────────────────────────────────


@router.get("/installed", response_model=List[SkillSummary])
async def list_installed(
    request: Request,
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> List[SkillSummary]:
    """List every skill the SkillManager can see, tagged with source."""
    mgr = await _get_scoped_manager(request, _user, db)
    summaries: list[SkillSummary] = []
    for skill in mgr._skills_cache.values():  # noqa: SLF001
        summaries.append(_skill_to_summary(skill))
    summaries.sort(key=lambda s: (s.source != "user", s.name.lower()))
    logger.info("Skill Hub: listed %d installed skill(s)", len(summaries))
    return summaries


@router.get("/installed/{name}", response_model=SkillDetail)
async def get_installed(
    name: str,
    request: Request,
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> SkillDetail:
    mgr = await _get_scoped_manager(request, _user, db)
    skill = await mgr.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return _skill_to_detail(skill)


@router.delete(
    "/installed/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_installed(
    name: str,
    request: Request,
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> Response:
    """Remove a user-installed skill. Builtin / external are refused."""
    mgr = await _get_scoped_manager(request, _user, db)
    skill = await mgr.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    source = _summary_source(skill)
    if source == "team":
        from xagent.skills.library import get_skill_write_provider

        writer = get_skill_write_provider()
        if writer is None:
            raise HTTPException(
                status_code=400, detail="No skill writer is registered for this scope."
            )
        try:
            await writer.delete_skill(
                _scope_context(request, _user, db),
                scope="team",
                name=name,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info("Skill Hub: deleted team skill %r", name)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    if source != "user":
        raise HTTPException(
            status_code=403,
            detail=(
                f"Cannot delete a {source} skill — only user-installed skills "
                "can be removed."
            ),
        )
    _delete_personal_skill(db=db, user=_user, name=name)
    logger.info("Skill Hub: deleted user skill %r", name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ──────────────────────────────────────────────────────────────────────
# Routes — in-UI authoring
# ──────────────────────────────────────────────────────────────────────


@router.post("/create", response_model=SkillSummary)
async def create_skill(
    body: CreateSkillRequest,
    request: Request,
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> SkillSummary:
    """Write a brand-new skill from in-UI input.

    The user supplies a name (used verbatim as the on-disk directory
    and the skill's external identifier) and the SKILL.md body. We
    refuse on duplicate names — overwrite via the edit endpoint is
    explicit, not implicit.
    """
    if body.scope != "personal":
        from xagent.skills.library import get_skill_write_provider

        writer = get_skill_write_provider()
        if writer is None:
            raise HTTPException(
                status_code=400, detail="No skill writer is registered for this scope."
            )
        try:
            await writer.create_skill(
                _scope_context(request, _user, db),
                scope=body.scope,
                name=body.name,
                files={"SKILL.md": body.skill_md.encode("utf-8")},
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        _write_personal_skill(
            db=db,
            user=_user,
            name=body.name,
            files={"SKILL.md": body.skill_md.encode("utf-8")},
        )

    mgr = await _get_scoped_manager(request, _user, db)
    skill = await mgr.get_skill(body.name)
    if skill is None:
        # Most likely cause: malformed YAML frontmatter that the parser
        # rejected. Leave the file on disk so the user can fix it via
        # PUT, but tell them why nothing showed up.
        raise HTTPException(
            status_code=400,
            detail=(
                "Skill written to disk but failed to re-parse — check the "
                "YAML frontmatter at the top of SKILL.md."
            ),
        )
    logger.info(
        "Skill Hub: created user skill %r (%d bytes)", body.name, len(body.skill_md)
    )
    return _skill_to_summary(skill)


@router.put("/installed/{name}", response_model=SkillSummary)
async def edit_installed(
    name: str,
    body: EditSkillRequest,
    request: Request,
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> SkillSummary:
    """Replace the SKILL.md of an installed user skill.

    Only ``user`` source is editable — builtin / external skills are
    refused so we don't silently fork a shipped skill (and so symlinked
    external roots stay readonly from our side).
    """
    mgr = await _get_scoped_manager(request, _user, db)
    skill = await mgr.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    source = _summary_source(skill)
    if source == "team":
        from xagent.skills.library import get_skill_write_provider

        writer = get_skill_write_provider()
        if writer is None:
            raise HTTPException(
                status_code=400, detail="No skill writer is registered for this scope."
            )
        try:
            await writer.update_skill_file(
                _scope_context(request, _user, db),
                scope="team",
                name=name,
                path="SKILL.md",
                content=body.skill_md.encode("utf-8"),
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    elif source != "user":
        raise HTTPException(
            status_code=403,
            detail="Only user-installed skills can be edited via the Hub.",
        )
    else:
        _update_personal_skill_md(db=db, user=_user, name=name, skill_md=body.skill_md)
    mgr = await _get_scoped_manager(request, _user, db)
    reloaded = await mgr.get_skill(name)
    if reloaded is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Edit written to disk but the parser rejected it. Fix the "
                "SKILL.md and PUT again — the bad version is still on disk."
            ),
        )
    logger.info("Skill Hub: edited user skill %r", name)
    return _skill_to_summary(reloaded)


# ──────────────────────────────────────────────────────────────────────
# Routes — registries list + registry proxy + install
# ──────────────────────────────────────────────────────────────────────


@router.get("/registries")
async def list_registries(
    _user: User = Depends(get_current_user),
) -> List[Dict[str, str]]:
    """Return available skill registries (ClawHub, etc.).
    The frontend uses this to build the source-selector dropdown."""
    return all_registries()


@router.get("/registry/list", response_model=RegistryListResponse)
async def registry_list(
    request: Request,
    sort: str = Query("installsCurrent"),
    limit: int = Query(24, ge=1, le=100),
    cursor: Optional[str] = Query(None),
    source: str = Query("clawhub"),
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> RegistryListResponse:
    """Browse a skill registry's catalog."""
    registry = get_registry(source)
    payload = await asyncio.to_thread(registry.list_skills, sort, limit, cursor)
    items_raw = payload.get("items", []) if isinstance(payload, dict) else []
    mgr = await _get_scoped_manager(request, _user, db)
    installed = _installed_slugs(mgr)
    items = [
        _summary_from_registry_item(i, installed, registry)
        for i in items_raw
        if isinstance(i, dict)
    ]
    next_cursor = payload.get("nextCursor") if isinstance(payload, dict) else None
    logger.info(
        "Skill Hub: registry/list source=%s sort=%s limit=%d → %d item(s), more=%s",
        source,
        sort,
        limit,
        len(items),
        "yes" if next_cursor else "no",
    )
    return RegistryListResponse(items=items, nextCursor=next_cursor)


@router.get("/registry/search", response_model=RegistryListResponse)
async def registry_search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(24, ge=1, le=100),
    source: str = Query("clawhub"),
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> RegistryListResponse:
    """Full-text search a skill registry."""
    registry = get_registry(source)
    payload = await asyncio.to_thread(registry.search_skills, q, limit)
    results_raw = (
        payload.get(registry.search_results_field, [])
        if isinstance(payload, dict)
        else []
    )
    mgr = await _get_scoped_manager(request, _user, db)
    installed = _installed_slugs(mgr)
    items = [
        _summary_from_registry_item(i, installed, registry)
        for i in results_raw
        if isinstance(i, dict)
    ]
    logger.info(
        "Skill Hub: registry/search source=%s q=%r → %d result(s)",
        source,
        q[:50],
        len(items),
    )
    return RegistryListResponse(items=items, nextCursor=None)


@router.post("/install/{source}", response_model=SkillSummary)
async def install_skill(
    source: str,
    body: InstallSkillRequest,
    request: Request,
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> SkillSummary:
    """Install a skill from a registry into ``~/.xagent/skills/<slug>/``."""
    _validate_skill_name(body.slug)

    # --- Look up registry ------------------------------------------
    registry = get_registry(source)

    # --- Scan + moderation gate ------------------------------------
    detail = await asyncio.to_thread(registry.get_skill, body.slug)
    if not isinstance(detail, dict):
        raise HTTPException(
            status_code=502,
            detail=f"{registry.display_name} detail had unexpected shape.",
        )
    _check_registry_security_gate(registry, detail)
    scan_status = registry.extract_scan_status(detail)

    # --- Download ZIP ----------------------------------------------
    dl_status, zip_bytes = await asyncio.to_thread(
        registry.download_skill, body.slug, body.version
    )
    if dl_status == 404:
        raise HTTPException(
            status_code=404,
            detail=f"{registry.display_name} skill or version not found.",
        )
    if dl_status >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"{registry.display_name} /download returned HTTP {dl_status}.",
        )
    if len(zip_bytes) > _MAX_DOWNLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Skill archive exceeds {_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MiB limit.",
        )

    # --- Store DB bundle -----------------------------------------
    files = _safe_zip_to_files(zip_bytes)
    if body.scope == "team":
        from xagent.skills.library import get_skill_write_provider

        writer = get_skill_write_provider()
        if writer is None:
            raise HTTPException(
                status_code=400, detail="No skill writer is registered for this scope."
            )
        try:
            await writer.create_skill(
                _scope_context(request, _user, db),
                scope="team",
                name=body.slug,
                files=files,
                origin=registry.id,
                metadata={
                    f"{registry.id}_slug": body.slug,
                    f"{registry.id}_version": body.version,
                },
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        _write_personal_skill(
            db=db,
            user=_user,
            name=body.slug,
            files=files,
            origin=registry.id,
            clawhub_slug=body.slug,
            clawhub_version=body.version,
        )

    mgr = await _get_scoped_manager(request, _user, db)
    skill = await mgr.get_skill(body.slug)
    if skill is None:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{registry.display_name} skill {body.slug!r} installed but failed "
                "to re-parse. Inspect SKILL.md by hand or remove and retry."
            ),
        )
    logger.info(
        "Skill Hub: installed %s skill %r (v%s, scan=%s)",
        registry.id,
        body.slug,
        body.version or "latest",
        scan_status,
    )
    return _skill_to_summary(skill)


@router.get("/registry/{slug}", response_model=RegistrySkillDetail)
async def registry_detail(
    slug: str,
    request: Request,
    source: str = Query("clawhub"),
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> RegistrySkillDetail:
    """Single-skill detail from a registry."""
    registry = get_registry(source)
    payload = await asyncio.to_thread(registry.get_skill, slug)
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected {registry.display_name} response shape.",
        )
    skill = payload.get("skill") or {}
    latest = payload.get("latestVersion") or {}
    moderation = payload.get("moderation")
    metadata = payload.get("metadata") or {}
    mgr = await _get_scoped_manager(request, _user, db)
    installed = _installed_slugs(mgr)
    return RegistrySkillDetail(
        slug=slug,
        displayName=str(skill.get("displayName") or skill.get("name") or slug),
        summary=str(skill.get("summary") or metadata.get("description") or ""),
        version=latest.get("version"),
        ownerHandle=(payload.get("owner") or {}).get("handle")
        or skill.get("ownerHandle"),
        homepage=metadata.get("homepage"),
        readme=metadata.get("readme")
        or latest.get("readme")
        or skill.get("description"),
        scanStatus=registry.extract_scan_status(payload),
        moderation=moderation if isinstance(moderation, dict) else None,
        installedAs=slug if slug in installed else None,
        registrySource=source,
        raw=payload,
    )
