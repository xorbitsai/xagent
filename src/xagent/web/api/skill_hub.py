"""Skill Hub API — manage user-installed skills (saas closed-source).

The Hub composes four capabilities on top of xagent's existing skill
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

  4. **File import** — install from an uploaded ``.zip`` skill folder or
     a bare ``SKILL.md`` (``POST /upload``). This is the only path that
     takes an archive straight from the client, so it bounds entry count,
     path length and content size before persisting, validates the bundle
     against the parser *before* naming it. Personal uploads return directly
     from those validated bytes after commit, so success does not depend on a
     fallible post-commit provider readback.

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
import functools
import hashlib
import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from xagent.skills.library import SkillScopeContext
from xagent.web.api.skill_hub_registry import (
    _MAX_DOWNLOAD_BYTES,
    SkillRegistry,
    all_registries,
    get_registry,
)
from xagent.web.auth_dependencies import get_current_user
from xagent.web.models.database import get_db
from xagent.web.models.user import User
from xagent.web.services.skill_runtime import (
    get_skill_runtime_scope,
    handoff_skill_runtime_session,
    invoke_skill_write_provider,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skill-hub", tags=["skill-hub"])


# ``CreateSkillRequest``/``EditSkillRequest`` cap ``skill_md`` at this many
# characters. SKILL.md is injected into the system prompt on every model turn,
# so an upload that bypassed the cap would push the context past what providers
# accept. Both write paths answer to the same number.
_MAX_SKILL_MD_CHARS = 200_000

# The longest a single character can be in UTF-8. Used to reject an oversized
# SKILL.md on its byte length, before anything is decoded.
_MAX_UTF8_BYTES_PER_CHAR = 4

# Characters that render as nothing but are *not* Unicode whitespace, so
# ``str.strip()`` leaves them: BOM / zero-width no-break space, zero-width
# space, and the zero-width non-joiner, joiner and word joiner. Stripped in
# addition to the default set, never instead of it -- passing an explicit
# argument to ``str.strip`` *replaces* the default, which silently dropped 21
# of the 29 characters ``isspace()`` recognizes, U+2003 EM SPACE among them.
_ZERO_WIDTH_CHARS = "\ufeff\u200b\u200c\u200d\u2060"


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
    skill_md: str = Field(..., min_length=1, max_length=_MAX_SKILL_MD_CHARS)
    scope: str = Field("personal", pattern="^(personal|team)$")


class EditSkillRequest(BaseModel):
    """``PUT /installed/{name}`` body. ``name`` is taken from the URL;
    only the SKILL.md content is mutable in v0."""

    skill_md: str = Field(..., min_length=1, max_length=_MAX_SKILL_MD_CHARS)


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


def _write_context(
    scope: SkillScopeContext,
) -> Any:
    from xagent.skills.library import SkillWriteContext

    return SkillWriteContext(
        user_id=scope.user_id,
        metadata=dict(scope.metadata),
    )


async def _get_scoped_manager(
    request: Request,
    context: SkillScopeContext,
    db: Any,
) -> Any:
    """Build a per-request SkillManager (no persistent per-user cache).

    Caching strategy — decouple by volatility:

    *Default path* (no custom provider registered):
      - Filesystem records (builtin / external / project skills) are stable, so
        we reuse the records already loaded by the process-wide
        ``app.state.skill_manager`` via ``StaticRecordsProvider``.
      - Personal-DB records are volatile, so ``XagentPersonalDbSkillProvider``
        opens its own short session and is queried fresh on every request.

    *Custom-provider path* (SaaS / overlay installed via
    ``set_skill_library_provider``): the provider is used as-is with the
    detached scope identity so that team-scoped records can be included.
    Each request still gets its own ``SkillManager`` instance, so there is no
    shared mutable state between concurrent requests.

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

    handoff_skill_runtime_session(db)

    custom_provider = get_skill_library_provider()
    if custom_provider is not None:
        # Custom (e.g. SaaS) provider — use as-is; it handles all layers.
        mgr = SkillManager(provider=custom_provider, context=context)
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
        mgr = SkillManager(provider=provider, context=context)

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


# Archiving a folder on macOS sweeps in Finder and resource-fork droppings.
# Any skill folder that has been opened in Finder carries them, so refusing a
# whole archive over a .DS_Store rejects an entirely ordinary bundle. They
# carry nothing a skill needs: drop them rather than fail the archive.
_IGNORED_ARCHIVE_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})

# Inflate archive members in slices so a decompression bomb cannot make the
# peak footprint a multiple of the size budget.
_ARCHIVE_CHUNK_BYTES = 1024 * 1024

# A skill bundle is a handful of documents. The byte budget alone does not bound
# the *count*: a 16 MiB archive of 100k empty padding directories passes it, and
# costs ~3s of CPU and ~62 MiB of allocation inside ``ZipFile(...)`` itself —
# before a single member is read. Cap the count, directories included: they are
# skipped for content but still cost a central-directory entry to walk, so
# excluding them would leave the cap evadable with padding folders. The same
# goes for AppleDouble cruft: a Finder-zipped bundle carries an "__MACOSX/._x"
# entry per file, so the cap is effectively halved for those. That is
# deliberate -- the cap is applied to what the archive actually contains,
# before anything knows which entries will be filtered out later.
_MAX_ARCHIVE_ENTRIES = 2000

# ``UserSkillFile.path`` is String(500). Beyond it PostgreSQL raises DataError
# at commit and the request dies as a 500; SQLite silently accepts, so this
# bound has to be enforced here rather than left to the database.
_MAX_SKILL_FILE_PATH_CHARS = 500


def _is_archive_cruft(path: str) -> bool:
    """True for OS bookkeeping files that are never part of a skill."""
    if path.startswith("__MACOSX/"):
        return True
    segments = path.split("/")
    return any(
        seg in _IGNORED_ARCHIVE_NAMES or seg.startswith("._") for seg in segments
    )


def _canonical_skill_path(raw_path: str) -> str:
    """Canonical POSIX form of a bundle path, or "" if nothing is left.

    One place decides what a path *is*, so root selection, containment and
    storage cannot disagree about it. Backslashes become separators (archives
    written on Windows use them), "." segments and repeated or leading slashes
    collapse. ".." is deliberately *not* resolved here: this returns a shape,
    and the caller rejects traversal rather than quietly resolving it into
    somewhere else.
    """
    segments = [
        seg for seg in str(raw_path).replace("\\", "/").split("/") if seg and seg != "."
    ]
    return "/".join(segments)


def _normalize_skill_files(
    files: dict[str, bytes], *, max_bytes: int | None = None
) -> dict[str, bytes]:
    # ``max_bytes`` lets a caller that advertised a smaller ceiling than the
    # shared one hold the *decompressed* bundle to the number it published.
    # Without it a lowered XAGENT_MAX_UPLOAD_SIZE bounded only the wire read,
    # and a compressed archive under that limit still expanded and persisted
    # up to the shared budget.
    budget = (
        _MAX_DOWNLOAD_BYTES
        if max_bytes is None
        else min(max_bytes, _MAX_DOWNLOAD_BYTES)
    )
    out: dict[str, bytes] = {}
    total = 0
    for raw_path, content in files.items():
        path = _canonical_skill_path(raw_path)
        if not path:
            # Nothing survived canonicalization ("///", ".", "./"). Refusing is
            # right, but it is not traversal, and saying so sent people looking
            # for a ".." that is not there.
            raise HTTPException(
                status_code=400,
                detail=f"Skill bundle contains an empty file path ({raw_path!r}).",
            )
        if ".." in path.split("/"):
            raise HTTPException(
                status_code=400,
                detail="Skill file path contains a path-traversal sequence.",
            )
        if _is_archive_cruft(path):
            continue
        # Check every segment, not just the first character of the whole path:
        # ".env" at the root was refused while "sub/.env" sailed through.
        dotted = next((seg for seg in path.split("/") if seg.startswith(".")), None)
        if dotted is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Skill bundle contains a hidden file ({dotted}). "
                    "Remove it and try again."
                ),
            )
        # Bound the path before the database has to. UserSkillFile.path is
        # String(500): PostgreSQL raises DataError at commit and the request
        # dies as a 500, while SQLite accepts it silently, so the check cannot
        # be left to the storage layer.
        if len(path) > _MAX_SKILL_FILE_PATH_CHARS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Skill file path is longer than "
                    f"{_MAX_SKILL_FILE_PATH_CHARS} characters: {path[:60]}…"
                ),
            )
        # Two distinct inputs can canonicalize to one path -- "./SKILL.md" and
        # "SKILL.md", or "a\\b.md" and "a/b.md". Writing both meant the last
        # one silently won and its neighbour vanished from a bundle the route
        # then reported as installed. Refuse instead: which of the two the
        # uploader meant is not ours to guess.
        if path in out:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Skill bundle contains two entries for {path!r}. "
                    "Remove the duplicate and try again."
                ),
            )
        total += len(content)
        if total > budget:
            raise HTTPException(
                status_code=413, detail="Skill files exceed size budget."
            )
        out[path] = bytes(content)
    if "SKILL.md" not in out:
        raise HTTPException(status_code=400, detail="Skill has no SKILL.md.")
    _check_skill_md_content(out["SKILL.md"])
    # Last gate before any caller writes. Everything above bounds the bundle's
    # shape; this asks the parser whether it can actually be loaded, so a
    # bundle that would fail on read-back is refused with a 400 naming the
    # file instead of being persisted and then surfacing as a post-write 500
    # with an orphaned row. Every write path -- registry install, personal
    # create, the migration loader -- reaches it through here.
    _assert_bundle_parses(out)
    return out


def _reject_oversized_text(content: bytes, label: str) -> None:
    """Refuse text past the character cap without decoding it.

    A UTF-8 character is at most four bytes, so anything longer than
    ``4 * _MAX_SKILL_MD_CHARS`` is over the cap whatever it decodes to, and no
    string has to be built to know it. The cost this avoids is real, not
    theoretical: a 40 KiB archive holding a 40 MiB member peaked at 120 MiB
    before being refused, because the decoded copy joined the extractor's
    chunks and joined bytes, all of it synchronous inside the install path.
    """
    if len(content) > _MAX_UTF8_BYTES_PER_CHAR * _MAX_SKILL_MD_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} is over {_MAX_SKILL_MD_CHARS} characters "
                f"({len(content)} bytes); the limit is {_MAX_SKILL_MD_CHARS}."
            ),
        )


# The only files ``SkillParser.parse_bundle`` decodes, in the order it reads
# them. Everything else in a bundle is carried as opaque bytes, so a decode
# failure can only have come from one of these. Kept in step with the parser
# by a test that reads parse_bundle's source, since a third decoded file added
# there would otherwise leave this list silently stale -- and silently skip
# the byte bound above.
_PARSER_DECODED_FILES = ("SKILL.md", "template.md")

# ``parse_bundle`` wants a name, but validation happens before naming matters
# and the value is never surfaced. Say so, so it reads as deliberate if it
# ever does reach a message.
_VALIDATION_PLACEHOLDER_NAME = "<bundle under validation>"


def _is_utf8(content: bytes) -> bool:
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _assert_bundle_parses(files: dict[str, bytes]) -> None:
    """Refuse a bundle that ``SkillManager.reload`` would fail to load.

    This is what makes the write safe to trust once it commits. ``reload``
    skips a record only when ``SkillParser.parse_bundle`` raises, so running
    that same call over the same bytes here matches its failure surface by
    construction rather than by restating the parser's rules -- including
    whatever the parser starts rejecting later.

    Today that surface is narrow: a missing ``SKILL.md``, and non-UTF-8 bytes
    in ``SKILL.md`` or ``template.md``, which are the only two files
    ``parse_bundle`` decodes. Malformed frontmatter is *not* in it --
    ``_extract_frontmatter`` swallows YAML errors -- so a bundle with broken
    YAML loads fine and must not be refused here. Checking more than the
    parser does would reject bundles ``reload`` would happily serve; the
    clauses below only translate what it raises into an HTTP response.

    Validating first is what lets the write be final. Committing and then
    re-reading would mean answering for a row that is already durable, and the
    only remedy at that point is a compensating delete -- which has to identify
    the row it is undoing, and gets it wrong whenever the name has been reused
    or the primary key recycled.
    """
    from xagent.skills.parser import SkillParser

    # Bound every file the parser will decode, not just SKILL.md: both are
    # decoded in full by ``parse_bundle``, so leaving template.md unbounded
    # reproduced the allocation spike SKILL.md was fixed for -- 120 MiB peak
    # for a 41 KiB archive.
    for path in _PARSER_DECODED_FILES:
        content = files.get(path)
        if content is not None:
            _reject_oversized_text(content, path)

    try:
        SkillParser.parse_bundle(name=_VALIDATION_PLACEHOLDER_NAME, files=files)
    except RecursionError as exc:
        # Deeply nested YAML blows the stack inside yaml.safe_load, which
        # ``_extract_frontmatter`` does not catch (it guards yaml.YAMLError).
        # Not a service defect: the input chose the depth, so it is a 400.
        raise HTTPException(
            status_code=400,
            detail="SKILL.md frontmatter is nested too deeply to parse.",
        ) from exc
    except UnicodeDecodeError as exc:
        # parse_bundle does not say which file failed, and that is the
        # actionable half of the message. Only the files the parser actually
        # decodes can be the cause: scanning every file instead would blame a
        # binary asset the parser never reads and hide the real culprit.
        culprit = next(
            (
                path
                for path in _PARSER_DECODED_FILES
                if path in files and not _is_utf8(files[path])
            ),
            None,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"{culprit} must be UTF-8 text."
                if culprit
                else "SKILL.md and template.md must be UTF-8 text."
            ),
        ) from exc
    except ValueError as exc:
        # The parser's own way of saying the bundle is unusable -- today, a
        # missing SKILL.md. Anything else propagates: a service defect must
        # keep its 5xx rather than be presented to the caller as bad input,
        # and nothing has been written yet, so letting it through cannot
        # leave a partial write behind.
        raise HTTPException(
            status_code=400, detail=f"Skill bundle could not be parsed: {exc}"
        ) from exc


def _is_skill_name_unique_violation(error: BaseException) -> bool:
    """Recognize the authoritative ``(user_id, name)`` unique-constraint failure.

    PostgreSQL names the constraint (``uq_user_skill_name``) in its message;
    SQLite names the columns (``user_skills.user_id, user_skills.name``).
    Matching either keeps an unrelated IntegrityError -- a foreign-key
    violation, or the ``(skill_id, path)`` file constraint -- from being
    reported as a duplicate name. Modelled on
    ``is_agent_name_unique_violation``, which solves the same problem for
    agents.
    """
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if "uq_user_skill_name" in message:
            return True
        if (
            "user_skills.user_id" in message
            and "user_skills.name" in message
            and ("unique" in message or "duplicate" in message)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _check_skill_md_content(skill_md: bytes) -> None:
    """Enforce the non-empty and character-count contract on SKILL.md.

    The byte bound runs first so nothing large is decoded to be rejected; past
    it the content is small by construction, so the remaining two questions
    are cheap to answer from text. ``errors="replace"`` keeps a non-UTF-8
    SKILL.md from raising here -- naming that file is the parser's job -- and a
    replacement character still occupies one character, so the cap holds
    either way.
    """
    _reject_oversized_text(skill_md, "SKILL.md")
    text = skill_md.decode("utf-8", errors="replace")
    # Match the non-empty contract Create/Edit enforce. A blank SKILL.md
    # persists happily and returns 200, then contributes nothing but a name to
    # every system prompt that loads it.
    #
    # Asked as a property of every character, not by stripping. Stripping only
    # peels the ends, so no fixed number of passes converges: three passes let
    # "\u200b \u200b \u200b" through, because each pass exposes the next
    # layer and the last one still leaves a zero-width character standing.
    # ``str.strip()`` also cannot be given the zero-width set as an argument --
    # that *replaces* the default rather than extending it, and drops 21 of the
    # 29 characters ``isspace()`` knows.
    if all(char.isspace() or char in _ZERO_WIDTH_CHARS for char in text):
        raise HTTPException(status_code=400, detail="SKILL.md is empty.")
    # Hold every write path to the character cap the Create/Edit schemas
    # already enforce. SKILL.md reaches the system prompt on every model turn,
    # so a bundle that skipped the cap would fail far from here, at inference
    # time. Counted in characters, not bytes, to match those schemas.
    char_count = len(text)
    if char_count > _MAX_SKILL_MD_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"SKILL.md is {char_count} characters; the limit is "
                f"{_MAX_SKILL_MD_CHARS}."
            ),
        )


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
    """Persist one personal skill, refusing anything that would not load.

    The bundle goes through ``_assert_bundle_parses`` before the commit -- by
    way of ``_normalize_skill_files`` below -- so a committed row is one the
    skill machinery can read and nothing here has to be undone afterwards. See
    that function for why undoing is the thing worth avoiding.
    """
    from xagent.skills.library import guess_media_type
    from xagent.web.models.skill import UserSkill, UserSkillFile

    _validate_skill_name(name)
    user_id = int(user.id)
    # Re-normalize rather than trusting the caller: this is the boundary the
    # registry install, personal create and migration paths all reach, and the
    # only place all three are guaranteed to be validated. The parse check the
    # commit depends on runs inside that call, so it is not repeated here --
    # it lives in the normalizer rather than in this writer because the team
    # scope reaches the write provider without passing through here at all.
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
    # The pre-check SELECT above and this write are not atomic: two concurrent
    # requests for the same (user, name) both see "no row" and both proceed.
    # The loser must still get the documented 409 rather than leaking an
    # IntegrityError as a 500. The guard spans the flush as well as the commit
    # because the INSERT reaches the database at flush time -- that is where
    # SQLite raises, while a deferred constraint would not surface until the
    # commit. Anything that is not this constraint is re-raised untouched.
    try:
        db.flush()
        # Read the key while the instance is guaranteed live. After the commit
        # the session may expire it, and re-reading would emit another SELECT
        # -- or fail outright once #1888 closes the session behind us.
        skill_id = int(skill.id)
        for path, content in sorted(normalized.items()):
            db.add(
                UserSkillFile(
                    skill_id=skill_id,
                    path=path,
                    content=content,
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                    media_type=guess_media_type(path),
                )
            )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        if not _is_skill_name_unique_violation(exc):
            raise
        raise HTTPException(
            status_code=409,
            detail=f"A personal skill named {name!r} already exists.",
        ) from exc


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
    """Delete one personal skill the caller owns, addressed by name."""
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


def _personal_summary_from_bundle(
    *, name: str, files: dict[str, bytes]
) -> SkillSummary:
    """Describe a personal skill from the validated bytes that were committed.

    The write is durable and was parsed before it landed, so the request
    succeeded. A replay with the same owner, name and bytes is reconciled to
    this same result, so transport retry cannot turn success into a false 409.

    The bytes re-parsed here are the ones just validated -- the same source
    the manager would have read -- and the result goes through
    ``_skill_to_summary`` like every other response, so this cannot drift away
    from the shape the normal path returns.
    """
    from xagent.skills.parser import SkillParser

    parsed = SkillParser.parse_bundle(name=name, files=files)
    parsed["scope"] = "personal"
    return _skill_to_summary(parsed)


def _personal_upload_matches_bundle(
    *, db: Any, user: User, name: str, files: dict[str, bytes]
) -> bool:
    """Whether an upload retry already produced this exact personal skill."""
    from sqlalchemy.orm import selectinload

    from xagent.web.models.skill import UserSkill

    existing = (
        db.query(UserSkill)
        .options(selectinload(UserSkill.files))
        .filter(
            UserSkill.user_id == int(user.id),
            UserSkill.name == name,
            UserSkill.origin == "upload",
        )
        .first()
    )
    if existing is None:
        return False
    existing_files = {item.path: bytes(item.content) for item in existing.files}
    return existing_files == files


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
    """Read a skill ZIP into a normalized skill file bundle."""
    files, _root = _safe_zip_extract(zip_bytes)
    return files


def _safe_zip_extract(
    zip_bytes: bytes, *, bad_zip_status: int = 502, max_bytes: int | None = None
) -> tuple[dict[str, bytes], str]:
    """Read a skill ZIP into ``(normalized files, root dir name)``.

    ``bad_zip_status`` is the status for an archive that cannot be read at
    all, letting the caller say who supplied it: 502 when it arrived from a
    registry proxy (the default), 4xx when the client handed it to us.
    Rejections about an archive's *contents* keep their own status
    regardless, so a caller cannot mask a traversal or a size overrun.

    The size budget is enforced on the *actual* decompressed byte count,
    not the sizes declared in the ZIP headers — a hostile archive can
    declare small sizes for members that inflate far larger.
    """
    # One guard around the whole "read an untrusted archive" region, rather than
    # naming exception types per call site. Enumerating them only ever covers
    # what was thought of: a tampered end-of-central-directory offset raises
    # ValueError from the constructor, an encrypted member RuntimeError, a
    # broken DEFLATE stream zlib.error, BZIP2/LZMA corruption OSError or
    # lzma.LZMAError — none of which subclass one another. Whatever zipfile
    # raises next is covered here too. Our own HTTPExceptions re-raise unchanged
    # so their specific status and message survive.
    budget = (
        _MAX_DOWNLOAD_BYTES
        if max_bytes is None
        else min(max_bytes, _MAX_DOWNLOAD_BYTES)
    )
    total = 0
    raw_files: dict[str, bytes] = {}
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        # Cap the entry count. This runs *after* ZipFile has materialized the
        # central directory, so it bounds the per-entry work that follows --
        # the ORM insert per file, the walk over every member -- but not the
        # construction itself. Bounding construction needs the archive's cost
        # to be limited before CPython parses it, which is a separate problem
        # with its own failure history; it is tracked in #1941 rather than
        # solved here. Measured, the worst archive this admits costs ~0.6s and
        # ~11 MiB inside the constructor before this rejects it.
        entries = zf.infolist()
        if len(entries) > _MAX_ARCHIVE_ENTRIES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Skill archive holds {len(entries)} entries; the limit is "
                    f"{_MAX_ARCHIVE_ENTRIES}."
                ),
            )
        for info in entries:
            if info.is_dir():
                continue
            # Canonicalize here, not at the tail. Root selection, the
            # containment check below and the normalizer all key off this
            # string, and they only agree if one function decides its shape:
            # "./SKILL.md" counts as depth 0 rather than losing to a nested
            # real root, and "my-skill\\notes.md" is seen as living under
            # "my-skill/" instead of being dropped as outside it.
            path = _canonical_skill_path(info.filename)
            if not path or ".." in path.split("/"):
                raise HTTPException(
                    status_code=400, detail="Skill ZIP contains unsafe paths."
                )
            # Distinct members can canonicalize to one path. Left alone the
            # later one overwrote the earlier, so a bundle came back missing a
            # file with a 200 attached. The normalizer refuses duplicates too;
            # this one has to exist as well because that check runs on the
            # already-collapsed dict, after the loss it is meant to catch.
            if path in raw_files:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Skill archive contains two entries for {path!r}. "
                        "Remove the duplicate and try again."
                    ),
                )
            remaining = budget - total
            # Reject on the declared size before inflating anything. zipfile
            # validates each member against its declared length and CRC while
            # reading, so a header that under-declares fails the read rather
            # than smuggling bytes past this check; the running total below
            # backstops it regardless.
            if info.file_size > remaining:
                raise HTTPException(
                    status_code=413, detail="Skill ZIP exceeds size budget."
                )
            # Inflate in bounded chunks rather than one read(remaining + 1):
            # a single read of the whole budget holds the result and zipfile's
            # own growing buffer at once, so a ~800 KiB bomb peaked at twice
            # the 50 MiB budget. Chunking caps the overshoot at one chunk.
            chunks: list[bytes] = []
            read_total = 0
            with zf.open(info) as member:
                while True:
                    chunk = member.read(_ARCHIVE_CHUNK_BYTES)
                    if not chunk:
                        break
                    read_total += len(chunk)
                    if read_total > remaining:
                        # Belt and braces, and deliberately not covered by a
                        # test: every way to get here is already intercepted,
                        # by the declared-size check above or by zipfile's own
                        # CRC/length validation of the member. It stands so a
                        # member that ever does out-produce its header -- a
                        # zipfile change, a format we do not decompress today
                        # -- cannot walk past the budget.
                        raise HTTPException(
                            status_code=413, detail="Skill ZIP exceeds size budget."
                        )
                    chunks.append(chunk)
            content = b"".join(chunks)
            total += len(content)
            raw_files[path] = content
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "Skill Hub: unreadable archive (%s)", type(exc).__name__, exc_info=True
        )
        raise HTTPException(
            status_code=bad_zip_status,
            detail="Skill archive is not a readable ZIP.",
        ) from exc

    # Choose the root by depth, not by alphabet. A plain sort asserted the
    # wrong invariant: a skill folder shipping its own "Examples/SKILL.md"
    # sorts before the real root marker, so the example was imported *as* the
    # skill — named "Examples", with the true root's files silently dropped and
    # a 200 returned. Cruft is filtered first so a crafted "__MACOSX/SKILL.md"
    # cannot win either.
    skill_md_paths = [
        path
        for path in raw_files
        if (path.endswith("/SKILL.md") or path == "SKILL.md")
        and not _is_archive_cruft(path)
    ]
    if not skill_md_paths:
        raise HTTPException(
            status_code=400, detail="Skill archive has no SKILL.md anywhere in it."
        )
    min_depth = min(path.count("/") for path in skill_md_paths)
    shallowest = sorted(p for p in skill_md_paths if p.count("/") == min_depth)
    if len(shallowest) > 1:
        # Two skills at the same depth: picking one would discard the other, so
        # make the user say which. Bounded so a crafted archive cannot reflect
        # an unlimited list of its own directory names back through the error.
        roots = ", ".join(
            (path.removesuffix("SKILL.md").rstrip("/") or ".")[:60]
            for path in shallowest[:5]
        )
        if len(shallowest) > 5:
            roots += f", and {len(shallowest) - 5} more"
        # 400 on both paths, matching the "no SKILL.md" rejection above:
        # ``bad_zip_status`` distinguishes who supplied an *unreadable* archive,
        # and this one parsed fine — it just carries the wrong contents.
        raise HTTPException(
            status_code=400,
            detail=(
                f"Skill archive contains multiple skills ({roots}). "
                "Upload one skill per archive."
            ),
        )
    skill_root = shallowest[0].removesuffix("SKILL.md").rstrip("/")
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
    return (
        _normalize_skill_files(files, max_bytes=max_bytes),
        skill_root.rsplit("/", 1)[-1],
    )


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
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
) -> List[SkillSummary]:
    """List every skill the SkillManager can see, tagged with source."""
    mgr = await _get_scoped_manager(request, context, db)
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
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
) -> SkillDetail:
    mgr = await _get_scoped_manager(request, context, db)
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
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> Response:
    """Remove a user-installed skill. Builtin / external are refused.

    ``name`` is resolved through the manager, so every provider keeps its own
    namespace here -- including names that look like an addressing scheme.
    ``:`` is not reserved by the provider contracts and the team create path
    does not run ``_validate_skill_name``, so a compliant provider may
    legitimately own a record named ``id:42``; parsing such a prefix here
    would delete the caller's unrelated personal row 42 instead.

    There is deliberately no by-primary-key counterpart: the column has no
    AUTOINCREMENT, so a client retrying a lost delete would remove whatever
    reused the id. A row the skill library cannot load is therefore stuck
    rather than recoverable -- tracked in #1911.
    """
    mgr = await _get_scoped_manager(request, context, db)
    skill = await mgr.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    source = _summary_source(skill)
    if source == "team":
        from xagent.skills.library import get_skill_write_provider

        await invoke_skill_write_provider(
            get_skill_write_provider(),
            "delete_skill",
            _write_context(context),
            scope="team",
            name=name,
        )
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
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
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

        await invoke_skill_write_provider(
            get_skill_write_provider(),
            "create_skill",
            _write_context(context),
            scope=body.scope,
            name=body.name,
            files={"SKILL.md": body.skill_md.encode("utf-8")},
        )
    else:
        _write_personal_skill(
            db=db,
            user=_user,
            name=body.name,
            files={"SKILL.md": body.skill_md.encode("utf-8")},
        )

    files = {"SKILL.md": body.skill_md.encode("utf-8")}
    mgr = await _get_scoped_manager(request, context, db)
    skill = await mgr.get_skill(body.name)
    if skill is None:
        if body.scope != "personal":
            # Provider-owned scopes validate inside their own writer, so an
            # absent record here is that provider's failure to report.
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Skill {body.name!r} was written to the {body.scope} scope "
                    "but the provider does not serve it back."
                ),
            )
        # The personal write is durable and was parsed before it landed, so
        # the request succeeded and the read side is merely behind. See
        # _personal_summary_from_bundle for why this is not a 5xx.
        logger.info(
            "Skill Hub: created user skill %r (%d bytes)",
            body.name,
            len(body.skill_md),
        )
        logger.warning(
            "Skill Hub: created user skill %r but the library does not serve "
            "it yet; answering from the validated bundle",
            body.name,
        )
        return _personal_summary_from_bundle(name=body.name, files=files)
    logger.info(
        "Skill Hub: created user skill %r (%d bytes)", body.name, len(body.skill_md)
    )
    return _skill_to_summary(skill)


@router.put("/installed/{name}", response_model=SkillSummary)
async def edit_installed(
    name: str,
    body: EditSkillRequest,
    request: Request,
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> SkillSummary:
    """Replace the SKILL.md of an installed user skill.

    Only ``user`` source is editable — builtin / external skills are
    refused so we don't silently fork a shipped skill (and so symlinked
    external roots stay readonly from our side).
    """
    mgr = await _get_scoped_manager(request, context, db)
    skill = await mgr.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    source = _summary_source(skill)
    if source == "team":
        from xagent.skills.library import get_skill_write_provider

        await invoke_skill_write_provider(
            get_skill_write_provider(),
            "update_skill_file",
            _write_context(context),
            scope="team",
            name=name,
            path="SKILL.md",
            content=body.skill_md.encode("utf-8"),
        )
    elif source != "user":
        raise HTTPException(
            status_code=403,
            detail="Only user-installed skills can be edited via the Hub.",
        )
    else:
        _update_personal_skill_md(db=db, user=_user, name=name, skill_md=body.skill_md)
    mgr = await _get_scoped_manager(request, context, db)
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


def _slugify_skill_name(raw: str) -> str:
    """Collapse arbitrary text into the [A-Za-z0-9_-]+ shape
    ``_validate_skill_name`` accepts; empty string if nothing survives."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", raw.strip()).strip("-_")[:64]


def _derive_upload_skill_name(
    filename: str, zip_root: str, skill_md: bytes, override: str | None = None
) -> str:
    """Pick a skill name for an uploaded bundle.

    Priority: caller-supplied ``override`` → ZIP root directory name (how
    Claude-style skill folders are usually zipped) → frontmatter ``name`` →
    upload filename stem.

    An override is the caller stating intent, so one that would be rewritten
    by slugification is refused instead: quietly turning "bad name!" into
    "bad-name" hands back a skill nobody asked for.
    """
    from xagent.skills.parser import SkillParser

    if override is not None and override != "":
        candidate = override
        # Validate against the rule the message quotes, not a slugifier
        # round-trip. The slugifier strips leading and trailing "-"/"_", so it
        # rejected names like "_foo" and "my_skill_" that _NAME_RE accepts and
        # POST /create takes — with an error citing the regex they do match.
        if not _NAME_RE.match(candidate) or len(candidate) > 64:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Skill name must match [A-Za-z0-9_-]+ and be at most 64 "
                    "characters; it is not rewritten for you."
                ),
            )
        return candidate

    frontmatter = SkillParser._extract_frontmatter(  # noqa: SLF001
        skill_md.decode("utf-8", errors="replace")
    )
    fm_name = frontmatter.get("name")
    candidates = [
        zip_root,
        fm_name if isinstance(fm_name, str) else "",
        Path(filename).stem,
    ]
    for raw in candidates:
        slug = _slugify_skill_name(raw)
        if slug:
            return slug
    raise HTTPException(
        status_code=400, detail="Could not derive a skill name from the upload."
    )


@router.post("/upload", response_model=SkillSummary)
async def upload_skill(
    request: Request,
    file: UploadFile = File(...),
    scope: str = Form("personal"),
    name: Optional[str] = Form(None),
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> SkillSummary:
    """Install a skill from an uploaded file.

    Accepts either a ``.zip`` skill bundle (a Claude-style skill folder
    with SKILL.md at its root, possibly nested one directory deep) or a
    bare ``.md`` file used verbatim as SKILL.md.

    The bundle is validated before the write, so a committed personal row is
    one the skill machinery can read and there is nothing to undo afterwards.
    Personal uploads answer directly from the validated bundle; provider-owned
    scopes keep their scoped manager readback contract.
    """
    if scope not in ("personal", "team"):
        raise HTTPException(
            status_code=400, detail="scope must be 'personal' or 'team'."
        )

    # Honour the deployment's configured upload limit, clamped to the budget
    # the extractor enforces on decompressed bytes: advertising a 100 MiB limit
    # while silently reading 50 MiB reports the wrong number in both
    # directions -- a smaller XAGENT_MAX_UPLOAD_SIZE was ignored, and a larger
    # one rejected uploads it claimed to allow.
    from xagent.config import get_max_upload_size_bytes

    limit = min(get_max_upload_size_bytes(), _MAX_DOWNLOAD_BYTES)
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {limit // (1024 * 1024)} MiB limit.",
        )

    filename = (file.filename or "").strip()
    lower = filename.lower()
    if lower.endswith(".zip"):
        # Off the event loop: inflating and normalizing an archive is pure CPU
        # over untrusted input, bounded but not free (thousands of members,
        # tens of MiB). Only this step moves -- it touches no ORM object, so
        # the request-bound Session stays on its own thread, which is why the
        # persistence below is deliberately left where it is.
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

    # Validate the bundle before anything reads it — naming included. Deriving
    # the name parses frontmatter itself, so leaving validation until after
    # would let an unparsable bundle raise from the naming step instead.
    # Validation must not depend on which naming branch a request happens to
    # take; that ordering coupling is what made the override path a bypass.
    # ``name`` only labels the parse error, so validating before the name is
    # known loses nothing and keeps the check independent of naming.
    files = _normalize_skill_files(files, max_bytes=limit)

    skill_name = _derive_upload_skill_name(
        filename, zip_root, files["SKILL.md"], override=name
    )

    if scope == "team":
        from xagent.skills.library import get_skill_write_provider

        await invoke_skill_write_provider(
            get_skill_write_provider(),
            "create_skill",
            _write_context(context),
            scope="team",
            name=skill_name,
            files=files,
            origin="upload",
        )
    else:
        try:
            _write_personal_skill(
                db=db,
                user=_user,
                name=skill_name,
                files=files,
                origin="upload",
            )
        except HTTPException as exc:
            if exc.status_code != 409 or not _personal_upload_matches_bundle(
                db=db,
                user=_user,
                name=skill_name,
                files=files,
            ):
                raise
            logger.info(
                "Skill Hub: replayed upload of personal skill %r from %r",
                skill_name,
                filename,
            )
        else:
            logger.info(
                "Skill Hub: uploaded skill %r from %r (%d bytes, %d file(s))",
                skill_name,
                filename,
                len(data),
                len(files),
            )
        # Personal bundles were parsed before the durable write, so their
        # authoritative response does not need a fallible manager/provider
        # readback after commit. Returning the same validated bytes also makes
        # an identical retry safe when the first response was lost.
        return _personal_summary_from_bundle(name=skill_name, files=files)

    mgr = await _get_scoped_manager(request, context, db)
    skill = await mgr.get_skill(skill_name)
    if skill is not None and _summary_source(skill) != (
        "user" if scope == "personal" else scope
    ):
        # ``reload`` keys the cache by name with filesystem records loaded
        # first, so a same-named builtin can stay resident and a bare
        # ``is None`` test would return 200 carrying that unrelated skill's
        # content. Treat a mismatch as "not visible yet", not as our write.
        logger.warning(
            "Skill Hub: %r read back as a %s skill, not the %s skill just "
            "written; answering from the validated bundle",
            skill_name,
            _summary_source(skill),
            scope,
        )
        skill = None
    if skill is None:
        if scope != "personal":
            # Provider-owned scopes validate inside their own writer, so an
            # absent record here is that provider's failure to report.
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Skill {skill_name!r} was written to the {scope} scope "
                    "but the provider does not serve it back."
                ),
            )
        # The personal write is durable and was parsed before it landed, so
        # the request succeeded and the read side is merely behind. No undo:
        # rolling back here is what made the old name-keyed compensation able
        # to delete a concurrent attempt's row. See
        # _personal_summary_from_bundle for why this is not a 5xx.
        logger.warning(
            "Skill Hub: uploaded skill %r but the library does not serve it "
            "yet; answering from the validated bundle",
            skill_name,
        )
        logger.info(
            "Skill Hub: uploaded skill %r from %r (%d bytes, %d file(s))",
            skill_name,
            filename,
            len(data),
            len(files),
        )
        return _personal_summary_from_bundle(name=skill_name, files=files)
    logger.info(
        "Skill Hub: uploaded skill %r from %r (%d bytes, %d file(s))",
        skill_name,
        filename,
        len(data),
        len(files),
    )
    return _skill_to_summary(skill)


# ──────────────────────────────────────────────────────────────────────
# Routes — registries list + registry proxy + install
# ──────────────────────────────────────────────────────────────────────


@router.get("/registries")
async def list_registries(
    _context: SkillScopeContext = Depends(get_skill_runtime_scope),
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
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
) -> RegistryListResponse:
    """Browse a skill registry's catalog."""
    registry = get_registry(source)
    payload = await asyncio.to_thread(registry.list_skills, sort, limit, cursor)
    items_raw = payload.get("items", []) if isinstance(payload, dict) else []
    mgr = await _get_scoped_manager(request, context, db)
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
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
) -> RegistryListResponse:
    """Full-text search a skill registry."""
    registry = get_registry(source)
    payload = await asyncio.to_thread(registry.search_skills, q, limit)
    results_raw = (
        payload.get(registry.search_results_field, [])
        if isinstance(payload, dict)
        else []
    )
    mgr = await _get_scoped_manager(request, context, db)
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
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
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

        await invoke_skill_write_provider(
            get_skill_write_provider(),
            "create_skill",
            _write_context(context),
            scope="team",
            name=body.slug,
            files=files,
            origin=registry.id,
            metadata={
                f"{registry.id}_slug": body.slug,
                f"{registry.id}_version": body.version,
            },
        )
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

    mgr = await _get_scoped_manager(request, context, db)
    skill = await mgr.get_skill(body.slug)
    if skill is None:
        if body.scope == "team":
            raise HTTPException(
                status_code=500,
                detail=(
                    f"{registry.display_name} skill {body.slug!r} was written to "
                    "the team scope but the provider does not serve it back."
                ),
            )
        logger.info(
            "Skill Hub: installed %s skill %r (v%s, scan=%s)",
            registry.id,
            body.slug,
            body.version or "latest",
            scan_status,
        )
        logger.warning(
            "Skill Hub: installed %r but the library does not serve it yet; "
            "answering from the validated bundle",
            body.slug,
        )
        return _personal_summary_from_bundle(name=body.slug, files=files)
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
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Any = Depends(get_db),
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
    mgr = await _get_scoped_manager(request, context, db)
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
