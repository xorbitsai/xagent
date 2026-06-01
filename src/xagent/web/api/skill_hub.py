"""Skill Hub API — manage user-installed skills (saas closed-source).

The Hub composes three capabilities on top of xagent's existing skill
machinery (``SkillManager`` + ``SkillParser``):

  1. **Local skill management** — list / detail / delete the skills
     currently visible to the SkillManager, tagging each with
     ``source`` (builtin / user / external) so the UI can gate
     destructive operations on user-installed skills only.

  2. **ClawHub registry browse & install** — a thin proxy in front of
     ``https://clawhub.ai/api/v1/*`` (the public, anonymous-readable
     OpenClaw skill registry). v0 install policy: ``scanStatus !=
     "clean"`` is refused server-side; never trust the client to honor
     a "are you sure?" prompt for malware.

  3. **In-UI authoring** — write a new SKILL.md from scratch
     (``POST /create``) or edit an installed one in place
     (``PUT /installed/{name}``). Edits and creates both invalidate
     the same cache the chat runtime reads from.

GitHub-URL import was removed in this iteration: we previously
shipped a ``git clone --depth=1`` path, but ClawHub gives us trusted
binaries with provenance and scan results, so we don't need to
re-implement that surface area. If someone really wants an
unscanned-source install path back, ``git`` is still on the box.

All writes land under ``<storage_root>/skills/`` (default
``~/.xagent/skills/``) — the same writable third root that
``skills/utils._get_default_skill_dirs`` configures, so installs
become visible to agents on the next task run after
``manager.reload()``.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skill-hub", tags=["skill-hub"])


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Per docs.openclaw.ai/clawhub/http-api — base for the v1 surface.
# A future ``.well-known/clawhub.json`` lookup could make this dynamic,
# but v0 hardcodes (no SaaS-side config plumbing needed yet).
CLAWHUB_BASE = "https://clawhub.ai"
CLAWHUB_API = f"{CLAWHUB_BASE}/api/v1"

# Hard cap on registry response bodies — the registry pages well under
# this, and we don't want a malformed upstream response to OOM us.
_MAX_REGISTRY_BODY = 2 * 1024 * 1024  # 2 MiB
_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MiB — well above any sane skill ZIP

# A request-scoped session would be nicer but requires app-state
# plumbing; module-level is fine for v0 traffic levels.
_HTTP = requests.Session()
_HTTP.headers.update(
    {
        # ClawHub doesn't gate on user-agent today, but identifying our
        # traffic helps them debug if anything breaks downstream.
        "User-Agent": "xagent-saas-skill-hub/0.1 (+https://github.com/xorbitsai/xagent)",
        "Accept": "application/json",
    }
)


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
    # Raw upstream blob for any UI bits we don't have a typed slot for
    # yet (provenance, capability tags, etc.). UI can poke at this for
    # secondary detail panels.
    raw: Dict[str, Any] = Field(default_factory=dict)


class RegistryListResponse(BaseModel):
    items: List[RegistrySkillSummary]
    nextCursor: Optional[str] = None


class InstallClawhubRequest(BaseModel):
    slug: str = Field(..., min_length=1, max_length=128)
    version: Optional[str] = None  # ClawHub default: latest


class RegistryStats(BaseModel):
    """Pagination metadata for the UI.

    ClawHub has no total-count endpoint, so we walk the cursor pages
    on the server (max 100 per round-trip via ClawHub's page-size cap)
    and sum up. Cached per-sort for a few minutes — the first call on
    a cold cache pays for the walk, subsequent calls return instantly.
    """

    sort: str
    total: int
    walked_pages: int  # how many ClawHub pages we actually fetched
    truncated: bool  # True if we hit the safety cap without exhausting cursors


class FeaturedSkill(RegistrySkillSummary):
    """A ClawHub skill we've editorially highlighted on the Discover
    tab, with our pitch attached."""

    featuredReason: str = ""


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


def _ensure_writable_target(name: str, *, allow_existing: bool = False) -> Path:
    """Resolve ``<user_skills_root>/<name>``, optionally allowing the
    directory to already exist (for edits). Always checks the resolved
    path can't escape the user root via symlinks."""
    _validate_skill_name(name)
    root = _user_skills_root().resolve()
    target = (root / name).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Resolved skill path escapes the user skills root.",
        ) from exc
    if target.exists() and not allow_existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"A skill named {name!r} already exists. Delete it first or rename."
            ),
        )
    return target


async def _get_manager(request: Request) -> Any:
    """Hand back the SkillManager singleton xagent put on app.state.
    All web/chat/agent paths share this instance; calling ``reload()``
    on it after a write updates everyone. Returns the live
    SkillManager; typed as ``Any`` to keep the skills package out of
    this module's import graph."""
    mgr = getattr(request.app.state, "skill_manager", None)
    if mgr is None:
        from xagent.skills.utils import create_skill_manager

        mgr = create_skill_manager()
        request.app.state.skill_manager = mgr
    await mgr.ensure_initialized()
    return mgr


def _skill_to_summary(skill_dict: dict) -> SkillSummary:
    return SkillSummary(
        name=skill_dict["name"],
        description=skill_dict.get("description", ""),
        when_to_use=skill_dict.get("when_to_use", ""),
        tags=skill_dict.get("tags", []),
        source=_classify_source(skill_dict.get("path", "")),
    )


def _skill_to_detail(skill_dict: dict) -> SkillDetail:
    return SkillDetail(
        name=skill_dict["name"],
        description=skill_dict.get("description", ""),
        when_to_use=skill_dict.get("when_to_use", ""),
        tags=skill_dict.get("tags", []),
        source=_classify_source(skill_dict.get("path", "")),
        content=skill_dict.get("content", ""),
        execution_flow=skill_dict.get("execution_flow", ""),
        files=skill_dict.get("files", []),
        path=skill_dict.get("path", ""),
    )


# ──────────────────────────────────────────────────────────────────────
# Helpers — ClawHub proxy
# ──────────────────────────────────────────────────────────────────────


def _clawhub_json(path: str, params: Optional[dict] = None) -> Any:
    """GET ``CLAWHUB_API/<path>`` and parse JSON. Translates upstream
    errors into HTTPExceptions with useful detail (never leaks raw
    upstream bodies — those can contain hostile content)."""
    url = f"{CLAWHUB_API}{path}"
    try:
        r = _HTTP.get(url, params=params or {}, timeout=15, stream=True)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"ClawHub unreachable: {exc}",
        ) from exc

    # Don't fully consume oversized bodies — guard against an
    # accidental gigabyte-of-JSON-from-upstream DoS.
    raw = r.raw.read(_MAX_REGISTRY_BODY + 1, decode_content=True)
    if len(raw) > _MAX_REGISTRY_BODY:
        raise HTTPException(status_code=502, detail="ClawHub response too large.")
    if r.status_code == 404:
        raise HTTPException(status_code=404, detail="Skill not found on ClawHub.")
    if r.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="ClawHub rate limit hit. Try again in a moment.",
        )
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"ClawHub returned HTTP {r.status_code}.",
        )
    try:
        return r.json() if not raw else __import__("json").loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not parse ClawHub JSON: {exc}"
        ) from exc


def _summary_from_registry_item(
    item: dict, installed_names: set[str]
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
        scanStatus=_extract_scan_status(item),
        installedAs=slug if slug in installed_names else None,
    )


def _extract_scan_status(item: dict) -> Optional[str]:
    """Pull ``security.status`` from wherever upstream put it (top
    level on detail responses, nested in ``latestVersion`` on some
    list shapes). Returns ``None`` for unscanned skills."""
    lv = item.get("latestVersion")
    if isinstance(lv, dict):
        sec = lv.get("security")
        if isinstance(sec, dict) and sec.get("status"):
            return str(sec["status"])
    sec = item.get("security")
    if isinstance(sec, dict) and sec.get("status"):
        return str(sec["status"])
    if item.get("scanStatus"):
        return str(item["scanStatus"])
    return None


def _installed_slugs(mgr: Any) -> set[str]:
    """Names of skills currently in the SkillManager cache. ClawHub
    slugs and local skill dir names line up because we install to
    ``<user_root>/<slug>/``, so a string-equal check is enough."""
    return set(mgr._skills_cache.keys())  # noqa: SLF001 — internal but stable


def _safe_extract_zip(zip_bytes: bytes, dest: Path) -> Path:
    """Extract a ClawHub skill ZIP into ``dest`` with path-safety.

    ClawHub ZIPs may wrap their content in a top-level directory
    (e.g. ``my-skill/SKILL.md``) or place ``SKILL.md`` at the root —
    we handle both. Returns the directory that contains SKILL.md.
    Raises HTTPException on:
      - bad zip / oversized members / path traversal
      - no SKILL.md anywhere in the archive
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=502, detail="ClawHub returned a bad ZIP."
        ) from exc

    dest_resolved = dest.resolve()
    # Pass 1: refuse if any entry escapes dest or is implausibly large.
    total = 0
    for info in zf.infolist():
        if info.file_size > _MAX_DOWNLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Skill ZIP member too large.")
        total += info.file_size
        if total > _MAX_DOWNLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail="Skill ZIP exceeds size budget."
            )
        target = (dest / info.filename).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Skill ZIP contains unsafe paths."
            ) from exc

    # Pass 2: actually extract.
    zf.extractall(dest)

    # Locate SKILL.md — accept root or one level deep.
    candidates = sorted(dest.rglob("SKILL.md"))
    if not candidates:
        raise HTTPException(
            status_code=400,
            detail="ClawHub artifact has no SKILL.md anywhere in it.",
        )
    return candidates[0].parent


# ──────────────────────────────────────────────────────────────────────
# Routes — local skills (list / detail / delete)
# ──────────────────────────────────────────────────────────────────────


@router.get("/installed", response_model=List[SkillSummary])
async def list_installed(
    request: Request,
    _user: User = Depends(get_current_user),
) -> List[SkillSummary]:
    """List every skill the SkillManager can see, tagged with source."""
    mgr = await _get_manager(request)
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
    _user: User = Depends(get_current_user),
) -> SkillDetail:
    mgr = await _get_manager(request)
    skill = await mgr.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return _skill_to_detail(skill)


@router.delete("/installed/{name}", status_code=204)
async def delete_installed(
    name: str,
    request: Request,
    _user: User = Depends(get_current_user),
) -> None:
    """Remove a user-installed skill. Builtin / external are refused."""
    mgr = await _get_manager(request)
    skill = await mgr.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    path = Path(skill.get("path", ""))
    source = _classify_source(str(path))
    if source != "user":
        raise HTTPException(
            status_code=403,
            detail=(
                f"Cannot delete a {source} skill — only user-installed skills "
                "can be removed."
            ),
        )
    # Symlink-escape defense: refuse if resolved path leaves user root.
    user_root = _user_skills_root().resolve()
    try:
        path.resolve().relative_to(user_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Refusing to delete: resolved path escapes the user skills root.",
        ) from exc
    shutil.rmtree(path, ignore_errors=False)
    await mgr.reload()
    logger.info("Skill Hub: deleted user skill %r", name)
    return None


# ──────────────────────────────────────────────────────────────────────
# Routes — in-UI authoring
# ──────────────────────────────────────────────────────────────────────


@router.post("/create", response_model=SkillSummary)
async def create_skill(
    body: CreateSkillRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> SkillSummary:
    """Write a brand-new skill from in-UI input.

    The user supplies a name (used verbatim as the on-disk directory
    and the skill's external identifier) and the SKILL.md body. We
    refuse on duplicate names — overwrite via the edit endpoint is
    explicit, not implicit.
    """
    target = _ensure_writable_target(body.name)
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(body.skill_md, encoding="utf-8")

    mgr = await _get_manager(request)
    await mgr.reload()
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
    _user: User = Depends(get_current_user),
) -> SkillSummary:
    """Replace the SKILL.md of an installed user skill.

    Only ``user`` source is editable — builtin / external skills are
    refused so we don't silently fork a shipped skill (and so symlinked
    external roots stay readonly from our side).
    """
    mgr = await _get_manager(request)
    skill = await mgr.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    path = Path(skill.get("path", ""))
    if _classify_source(str(path)) != "user":
        raise HTTPException(
            status_code=403,
            detail="Only user-installed skills can be edited via the Hub.",
        )
    # Symlink-escape defense: same check as delete.
    user_root = _user_skills_root().resolve()
    try:
        path.resolve().relative_to(user_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Refusing to edit: resolved path escapes the user skills root.",
        ) from exc

    (path / "SKILL.md").write_text(body.skill_md, encoding="utf-8")
    await mgr.reload()
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
# Routes — ClawHub registry proxy + install
# ──────────────────────────────────────────────────────────────────────


@router.get("/registry/list", response_model=RegistryListResponse)
async def registry_list(
    request: Request,
    sort: str = Query("trending"),
    limit: int = Query(24, ge=1, le=100),
    cursor: Optional[str] = Query(None),
    _user: User = Depends(get_current_user),
) -> RegistryListResponse:
    """Browse the ClawHub catalog. ``sort`` mirrors upstream's
    documented values (``trending`` / ``newest`` / ``installs`` / etc).
    Returns whatever ClawHub returns, plus ``installedAs`` so the UI
    can mark already-installed skills.

    Note: we deliberately do NOT pass ``nonSuspiciousOnly=true`` here.
    ClawHub interprets that flag as "only skills whose scan
    *positively cleared*", but with current registry-wide scan coverage
    near 0% it filters out essentially every result. Install-time
    gating (blacklist on malicious / quarantined / revoked) is what
    actually protects users; pre-filtering the list would just give
    them an empty Hub.
    """
    params = {"sort": sort, "limit": limit}
    if cursor:
        params["cursor"] = cursor
    payload = _clawhub_json("/skills", params)
    items_raw = payload.get("items", []) if isinstance(payload, dict) else []
    mgr = await _get_manager(request)
    installed = _installed_slugs(mgr)
    items = [
        _summary_from_registry_item(i, installed)
        for i in items_raw
        if isinstance(i, dict)
    ]
    next_cursor = payload.get("nextCursor") if isinstance(payload, dict) else None
    logger.info(
        "Skill Hub: registry/list sort=%s limit=%d cursor=%s → %d item(s), more=%s",
        sort,
        limit,
        "yes" if cursor else "none",
        len(items),
        "yes" if next_cursor else "no",
    )
    return RegistryListResponse(items=items, nextCursor=next_cursor)


@router.get("/registry/search", response_model=RegistryListResponse)
async def registry_search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(24, ge=1, le=100),
    _user: User = Depends(get_current_user),
) -> RegistryListResponse:
    """Full-text search ClawHub. Same response shape as ``/list``.
    See ``registry_list`` for why ``nonSuspiciousOnly`` is omitted."""
    params = {"q": q, "limit": limit}
    payload = _clawhub_json("/search", params)
    results_raw = payload.get("results", []) if isinstance(payload, dict) else []
    mgr = await _get_manager(request)
    installed = _installed_slugs(mgr)
    items = [
        _summary_from_registry_item(i, installed)
        for i in results_raw
        if isinstance(i, dict)
    ]
    logger.info("Skill Hub: registry/search q=%r → %d result(s)", q[:50], len(items))
    return RegistryListResponse(items=items, nextCursor=None)


@router.get("/registry/{slug}", response_model=RegistrySkillDetail)
async def registry_detail(
    slug: str,
    request: Request,
    _user: User = Depends(get_current_user),
) -> RegistrySkillDetail:
    """Single-skill detail from ClawHub. Bundles whatever upstream
    exposes (skill / latestVersion / metadata / moderation) into a
    flat shape the UI can render directly."""
    payload = _clawhub_json(f"/skills/{slug}")
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502, detail="Unexpected ClawHub response shape."
        )
    skill = payload.get("skill") or {}
    latest = payload.get("latestVersion") or {}
    moderation = payload.get("moderation")
    metadata = payload.get("metadata") or {}
    mgr = await _get_manager(request)
    installed = _installed_slugs(mgr)
    return RegistrySkillDetail(
        slug=slug,
        displayName=str(skill.get("displayName") or skill.get("name") or slug),
        summary=str(skill.get("summary") or metadata.get("description") or ""),
        version=latest.get("version"),
        ownerHandle=(payload.get("owner") or {}).get("handle")
        or skill.get("ownerHandle"),
        homepage=metadata.get("homepage"),
        readme=metadata.get("readme") or latest.get("readme"),
        scanStatus=(latest.get("security") or {}).get("status"),
        moderation=moderation if isinstance(moderation, dict) else None,
        installedAs=slug if slug in installed else None,
        raw=payload,
    )


@router.post("/install/clawhub", response_model=SkillSummary)
async def install_clawhub(
    body: InstallClawhubRequest,
    request: Request,
    _user: User = Depends(get_current_user),
) -> SkillSummary:
    """Install a ClawHub skill into ``~/.xagent/skills/<slug>/``.

    v0 install policy is a **blacklist**, not a whitelist:

    - ``scan_status == "malicious"`` → refused (403)
    - ``moderation.moderationState in {"quarantined", "revoked"}``
      → refused (403)
    - ``"clean"``, ``"suspicious"``, or ``None`` (unscanned) → allowed

    Whitelist mode (``"clean"`` only) was the original intent, but
    upstream's scan coverage is currently ~0% on trending skills, so
    a strict whitelist would block every install. The UI surfaces
    ``scanStatus`` via badge so users can see "Not scanned" before
    they hit Install.

    Flow:
      1. Validate slug + reserve target dir.
      2. Pull ClawHub detail to check scan + moderation. Refuse on
         the blacklist conditions above.
      3. Stream the ZIP from ``/download`` into memory (cap 50 MiB).
      4. Path-safe extract → locate SKILL.md → move into place.
      5. ``manager.reload()`` so the new skill is visible to agents.
    """
    _validate_skill_name(body.slug)
    target = _ensure_writable_target(body.slug)

    # --- 2. Scan + moderation gate ---------------------------------
    detail = _clawhub_json(f"/skills/{body.slug}")
    if not isinstance(detail, dict):
        raise HTTPException(
            status_code=502, detail="ClawHub detail had unexpected shape."
        )
    scan_status = _extract_scan_status(detail)
    moderation = detail.get("moderation") or {}
    moderation_state = (
        moderation.get("moderationState") if isinstance(moderation, dict) else None
    )
    if scan_status == "malicious":
        raise HTTPException(
            status_code=403,
            detail="Install refused: this skill is flagged malicious by ClawHub scanners.",
        )
    if moderation_state in ("quarantined", "revoked"):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Install refused: skill is {moderation_state} by ClawHub moderators."
            ),
        )

    # --- 3. Download ZIP -------------------------------------------
    dl_params = {"slug": body.slug}
    if body.version:
        dl_params["version"] = body.version
    try:
        dl = _HTTP.get(
            f"{CLAWHUB_API}/download",
            params=dl_params,
            timeout=60,
            stream=True,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502, detail=f"ClawHub download failed: {exc}"
        ) from exc
    if dl.status_code == 404:
        raise HTTPException(
            status_code=404, detail="ClawHub skill or version not found."
        )
    if dl.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"ClawHub /download returned HTTP {dl.status_code}.",
        )
    zip_bytes = dl.raw.read(_MAX_DOWNLOAD_BYTES + 1, decode_content=True)
    if len(zip_bytes) > _MAX_DOWNLOAD_BYTES:
        raise HTTPException(status_code=413, detail="ClawHub artifact too large.")

    # --- 4. Extract + move ------------------------------------------
    # Extract into a sibling staging dir, then promote it on success
    # so a half-written install doesn't end up live on disk.
    staging = target.parent / f".{body.slug}.installing"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        skill_root = _safe_extract_zip(zip_bytes, staging)
        if skill_root != staging:
            # ZIP wrapped the skill in a sub-dir — promote that sub-dir.
            tmp = target.parent / f".{body.slug}.promote"
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.move(str(skill_root), str(tmp))
            shutil.rmtree(staging, ignore_errors=True)
            shutil.move(str(tmp), str(target))
        else:
            shutil.move(str(staging), str(target))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # --- 5. Reload + return -----------------------------------------
    mgr = await _get_manager(request)
    await mgr.reload()
    skill = await mgr.get_skill(body.slug)
    if skill is None:
        raise HTTPException(
            status_code=500,
            detail=(
                f"ClawHub skill {body.slug!r} installed at {target} but failed "
                "to re-parse. Inspect SKILL.md by hand or remove and retry."
            ),
        )
    logger.info(
        "Skill Hub: installed ClawHub skill %r (v%s, scan=%s)",
        body.slug,
        body.version or "latest",
        scan_status,
    )
    return _skill_to_summary(skill)


# ──────────────────────────────────────────────────────────────────────
# Routes — featured ("Editor's pick" rail)
# ──────────────────────────────────────────────────────────────────────

# Editorial list of slugs xagent recommends on the Discover tab,
# alongside our reason for each. Sibling to this module — same dir as
# api/, one level up to xagent_saas package root.
_FEATURED_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "skill_hub_featured.json"
)

# Cheap in-memory cache: every request hitting /featured triggers N
# ClawHub detail calls (one per slug). Cache the joined result for a
# few minutes so the rail doesn't N-amplify on every page load.
# Refresh granularity vs. ClawHub freshness is the trade-off; 5 min
# means edits to the featured JSON take up to 5 min to surface, which
# matches "drop a file, no restart" + an acceptable update lag.
_FEATURED_CACHE: Dict[str, Any] = {"items": None, "fetched_at": 0.0}
_FEATURED_TTL_SECONDS = 300


def _load_featured_config() -> List[Dict[str, str]]:
    """Read the editorial featured list off disk every call. Cheap
    (small JSON, no DB), keeps "drop a file" workflow alive without
    restart. Returns ``[]`` on any read/parse error so a typo in the
    JSON degrades to "no featured rail" rather than 500ing the
    Discover tab."""
    if not _FEATURED_CONFIG_PATH.exists():
        return []
    try:
        data = json.loads(_FEATURED_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("skill_hub_featured.json unreadable: %s", exc)
        return []
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        out.append({"slug": slug, "reason": str(entry.get("reason") or "")})
    return out


@router.get("/featured", response_model=List[FeaturedSkill])
async def featured(
    request: Request,
    _user: User = Depends(get_current_user),
) -> List[FeaturedSkill]:
    """Editor's pick rail for the Discover tab.

    Reads the slug + reason list from ``skill_hub_featured.json``,
    fans out a ClawHub detail call per slug to enrich into the same
    shape as the registry list response (plus ``featuredReason``).
    Cached for 5 minutes — edits to the JSON file take up to that
    long to surface, but the rail won't N-amplify ClawHub per page
    load.

    Slugs that ClawHub doesn't know are silently dropped: the rail
    is editorial, not a contract, so one stale entry shouldn't kill
    the whole row.
    """
    config = _load_featured_config()
    if not config:
        return []

    now = time.time()
    cached = _FEATURED_CACHE.get("items")
    if (
        cached is not None
        and (now - _FEATURED_CACHE["fetched_at"]) < _FEATURED_TTL_SECONDS
    ):
        # Hot path. Just refresh the installedAs marker from the
        # live SkillManager so newly-installed skills flip the badge
        # without waiting for the TTL.
        mgr = await _get_manager(request)
        installed = _installed_slugs(mgr)
        return [
            FeaturedSkill(
                **{**it, "installedAs": it["slug"] if it["slug"] in installed else None}
            )
            for it in cached
        ]

    mgr = await _get_manager(request)
    installed = _installed_slugs(mgr)

    # Fan out the 6 detail fetches in parallel. ``_clawhub_json`` is
    # sync (uses the requests library), so we wrap each in
    # ``asyncio.to_thread`` to run on the default executor and gather
    # the results. Total wall time becomes max(per-call) instead of
    # sum(per-call) — typically 500ms vs 3s.
    async def _fetch(slug: str) -> Optional[Dict[str, Any]]:
        try:
            return await asyncio.to_thread(_clawhub_json, f"/skills/{slug}")
        except HTTPException as exc:
            logger.info("Skill Hub featured: dropping %r — %s", slug, exc.detail)
            return None

    details = await asyncio.gather(*[_fetch(e["slug"]) for e in config])

    items: list[dict] = []
    for entry, detail in zip(config, details):
        if not isinstance(detail, dict):
            continue
        slug = entry["slug"]
        skill = detail.get("skill") or {}
        latest = detail.get("latestVersion") or {}
        stats = skill.get("stats") or {}
        items.append(
            {
                "slug": slug,
                "displayName": str(skill.get("displayName") or slug),
                "summary": str(skill.get("summary") or ""),
                "version": latest.get("version")
                or (skill.get("tags") or {}).get("latest"),
                "ownerHandle": (detail.get("owner") or {}).get("handle"),
                "installs": stats.get("installsCurrent"),
                "updatedAt": skill.get("updatedAt"),
                "scanStatus": _extract_scan_status(detail),
                "installedAs": None,  # filled per-request from live manager
                "featuredReason": entry["reason"],
            }
        )

    _FEATURED_CACHE["items"] = items
    _FEATURED_CACHE["fetched_at"] = now
    return [
        FeaturedSkill(
            **{**it, "installedAs": it["slug"] if it["slug"] in installed else None}
        )
        for it in items
    ]


# ──────────────────────────────────────────────────────────────────────
# Routes — registry pagination stats
# ──────────────────────────────────────────────────────────────────────

# Per-sort cache of (total, walked_pages, truncated, fetched_at). We
# walk ClawHub's cursor pages on the server because there's no count
# endpoint upstream; the walk is sequential by nature (the next cursor
# depends on the previous response). Cached aggressively so the
# expensive first call amortizes.
_STATS_CACHE: Dict[str, Dict[str, Any]] = {}
_STATS_TTL_SECONDS = 300
# Stats walks ClawHub at its maximum supported page size to keep
# total round-trips low. The frontend's UI page size is 30 (see
# ``PAGE_SIZE`` in page.tsx) and ``totalPages`` is computed as
# ``ceil(total / 30)`` — the two are deliberately decoupled.
#
# Edge case: ClawHub's ``sort=trending`` returns "top N based on
# request limit, no pagination", so a walk at limit=100 sees 100
# items but a list at limit=30 also sees 30 + nextCursor=null. The
# UI then shows "Page 1 of 4" but Next is disabled — acceptable
# rough edge for the speedup; default sort is ``installsCurrent``
# (which paginates cleanly) so most users never hit it.
# Pagination math intentionally accepts "≥N" instead of exact count.
#
# ClawHub's full installsCurrent corpus is ~8,000 skills. Walking
# that exhaustively at page size 100 = 80 sequential round-trips ×
# ~1.4s/call to ClawHub = ~2 minutes wall time. That cost can't be
# parallelized (each cursor depends on the previous response), so a
# faithful total is structurally slow.
#
# Instead we walk 15 pages = 1,500 skills, then flag the result as
# truncated. UI renders "Page X of ≥50" — directionally honest
# ("there's lots") without the 2-minute prewarm. If a user really
# does paginate past page 50, the live ``Next`` button continues
# working (it uses the actual cursor from /list, not the stats cap).
_STATS_PAGE_SIZE = 100
_STATS_MAX_PAGES = 15


def _fill_stats_cache(sort: str) -> Dict[str, Any]:
    """Walk ClawHub for one sort dimension, populate ``_STATS_CACHE``,
    return the cache entry.

    Synchronous on purpose so it composes cleanly with both the
    HTTP endpoint (called via ``asyncio.to_thread``) and the
    startup prewarm task (which spawns a few of these in parallel).
    """
    total = 0
    pages = 0
    cursor: Optional[str] = None
    truncated = False
    hit_cap = True
    for _ in range(_STATS_MAX_PAGES):
        params: Dict[str, Any] = {"sort": sort, "limit": _STATS_PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        try:
            payload = _clawhub_json("/skills", params)
        except HTTPException as exc:
            # ClawHub flaked partway through the walk. Don't 502 the
            # whole stats call — partial total + ``truncated=True`` is
            # still useful to the UI (it'll render "Page N of ≥M").
            logger.warning(
                "Skill Hub: stats walk for sort=%s aborted at page %d (%s); "
                "returning partial total %d",
                sort,
                pages,
                exc.detail,
                total,
            )
            truncated = True
            break
        if not isinstance(payload, dict):
            break
        items = payload.get("items") or []
        total += len(items)
        pages += 1
        cursor = payload.get("nextCursor")
        if not cursor:
            hit_cap = False
            break
    if hit_cap and cursor:
        truncated = True

    entry = {
        "total": total,
        "walked_pages": pages,
        "truncated": truncated,
        "fetched_at": time.time(),
    }
    _STATS_CACHE[sort] = entry
    logger.info(
        "Skill Hub: stats cache filled sort=%s → %d skill(s) across %d page(s)%s",
        sort,
        total,
        pages,
        " (truncated)" if truncated else "",
    )
    return entry


@router.get("/registry/stats", response_model=RegistryStats)
async def registry_stats(
    sort: str = Query("trending"),
    _user: User = Depends(get_current_user),
) -> RegistryStats:
    """Walk ClawHub for the current ``sort`` until cursor exhausts,
    return the total skill count. Used by the UI to render "Page N
    of M" — without this endpoint the count side is unknowable from
    a cursor-based API.

    First-call cost (cold cache): ``ceil(total / 30)`` sequential
    round-trips to ClawHub. For ``sort=trending`` (capped at ~30 per
    page-size constraint) this is one call. For ``sort=newest``
    (~2,000) it's ~67 calls, ~5 seconds without prewarm. Backend
    startup runs ``prewarm`` to fill the common sorts ahead of time,
    so a typical user hits a warm cache.
    """
    cached = _STATS_CACHE.get(sort)
    now = time.time()
    if cached and (now - cached["fetched_at"]) < _STATS_TTL_SECONDS:
        return RegistryStats(
            sort=sort,
            total=cached["total"],
            walked_pages=cached["walked_pages"],
            truncated=cached["truncated"],
        )
    # Cold (or expired) — walk on a thread so we don't block the
    # event loop for the multi-second sequential cursor walk.
    entry = await asyncio.to_thread(_fill_stats_cache, sort)
    return RegistryStats(
        sort=sort,
        total=entry["total"],
        walked_pages=entry["walked_pages"],
        truncated=entry["truncated"],
    )


# ──────────────────────────────────────────────────────────────────────
# Startup prewarm
# ──────────────────────────────────────────────────────────────────────

# Sorts the Discover tab's "Page N of M" math depends on. Prewarming
# these lets a user's first hit on Discover skip the cold-cache walk
# (which is 3-5s per sort). Other sorts (stars / updated / newest)
# still cold-load on first user request, but they're less common.
_PREWARM_SORTS = ("installsCurrent", "trending")


async def prewarm() -> None:
    """Best-effort prewarm of common Skill Hub caches.

    Runs as a background task on backend startup; failures are
    logged but don't propagate. After this completes a user opening
    /skill-hub for the first time gets ~2.5s instead of ~9s before
    the rails populate.

    Prewarms (in parallel):
      - Featured rail (the editorial 6-slug fan-out)
      - Registry stats for the default sort + a likely second sort
    """
    logger.info("Skill Hub: starting prewarm…")
    t0 = time.time()

    async def _warm_featured() -> None:
        config = _load_featured_config()
        if not config:
            return

        async def _fetch_one(slug: str) -> Optional[Dict[str, Any]]:
            try:
                return await asyncio.to_thread(_clawhub_json, f"/skills/{slug}")
            except HTTPException:
                return None

        details = await asyncio.gather(*[_fetch_one(e["slug"]) for e in config])
        items: list[dict] = []
        for entry, detail in zip(config, details):
            if not isinstance(detail, dict):
                continue
            skill = detail.get("skill") or {}
            latest = detail.get("latestVersion") or {}
            stats = skill.get("stats") or {}
            items.append(
                {
                    "slug": entry["slug"],
                    "displayName": str(skill.get("displayName") or entry["slug"]),
                    "summary": str(skill.get("summary") or ""),
                    "version": latest.get("version")
                    or (skill.get("tags") or {}).get("latest"),
                    "ownerHandle": (detail.get("owner") or {}).get("handle"),
                    "installs": stats.get("installsCurrent"),
                    "updatedAt": skill.get("updatedAt"),
                    "scanStatus": _extract_scan_status(detail),
                    "installedAs": None,
                    "featuredReason": entry["reason"],
                }
            )
        _FEATURED_CACHE["items"] = items
        _FEATURED_CACHE["fetched_at"] = time.time()

    async def _warm_stats(sort: str) -> None:
        try:
            await asyncio.to_thread(_fill_stats_cache, sort)
        except Exception as exc:
            logger.warning("Skill Hub prewarm: stats walk for %s failed: %s", sort, exc)

    try:
        await asyncio.gather(
            _warm_featured(),
            *[_warm_stats(s) for s in _PREWARM_SORTS],
            return_exceptions=True,
        )
        logger.info("Skill Hub: prewarm done in %.1fs", time.time() - t0)
    except Exception as exc:
        logger.warning("Skill Hub: prewarm aborted: %s", exc)
