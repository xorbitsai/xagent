"""Validate and repair model-authored file links before they reach the UI."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import or_
from sqlalchemy.orm import Session

from ...core.file_ref import build_file_id_ref, parse_file_id_ref
from ...core.tools.artifacts import artifact_type_for_filename
from ..models.uploaded_file import UploadedFile

logger = logging.getLogger(__name__)

# The title clause supports all three CommonMark delimiters (``"..."``,
# ``'...'``, and ``(...)``) and backslash-escapes within them, so a
# well-formed pre-existing title (however it was written) still lets the
# base ``[label](file:id ...)`` match and go through id validation/repair
# below, rather than silently bypassing it. Every title/junk alternative
# also excludes ``)``, ``[``, and control characters:
#
# - ``)`` bounds a malformed title's runaway match to the current link's
#   own closing paren, instead of spanning past it into unrelated content.
# - ``[`` stops a title/junk span from ever crossing into what looks like
#   the START of another ``[label](file:...)`` reference. Without this, a
#   malformed first title can swallow a second, well-formed reference
#   whole (verified: ``[a](file:id1 "x [b](file:id2 ")`` collapses into
#   one match spanning both, silently dropping id2's reference from the
#   output). With it, the malformed first fragment simply fails to match
#   (left inert, not further mangled) and the second reference is still
#   recovered as its own independent match.
# - Control characters block the same paragraph-splitting hazard as
#   ``_UNSAFE_LABEL_RE``/``_UNSAFE_TITLE_RE`` below.
#
# A title clause that doesn't parse as any of the three forms (duplicated,
# unterminated, mismatched delimiters) falls through to the bare
# alternative, which discards it as unstructured trailing content up to
# the next ``)`` -- the reference is still validated, just without a title.
#
# This is the canonical parser for this ``[label](file:id ["title"])``
# format -- ``reconcile_assistant_file_references`` is what actually
# introduces the title clause into content. Other ``file:``-link-matching
# regexes elsewhere in the codebase (task_execution_context_service.py,
# websocket.py, slack/telegram channel utils) predate title support and
# are title-blind; none currently run on reconciled/titled content, but a
# future change piping reconciled text into one of them should update it
# to match this pattern (or route through a shared parsing helper) rather
# than silently mis-parsing a title clause as part of the id/label.
_MARKDOWN_FILE_REFERENCE_RE = re.compile(
    r"(?P<image>!)?\[(?P<label>[^\]]*)\]\("
    r"(?P<target>file:[^)\s]+)"
    r"(?:"
    r'(?:\s+(?:"(?P<dtitle>(?:[^"\\)[\x00-\x1f\x7f]|\\.)*)"'
    r"|'(?P<stitle>(?:[^'\\)[\x00-\x1f\x7f]|\\.)*)'"
    r"|\((?P<ptitle>(?:[^()\\[\x00-\x1f\x7f]|\\.)*)\)))?"
    r"|[^)[\x00-\x1f\x7f]*"
    r")"
    r"\)"
)


def _parsed_title(match: re.Match[str]) -> str | None:
    """The raw title text from whichever of the three delimiter forms matched."""
    dtitle = match.group("dtitle")
    if dtitle is not None:
        return dtitle
    stitle = match.group("stitle")
    if stitle is not None:
        return stitle
    return match.group("ptitle")


# ``artifact_type_for_filename`` only knows image/video/office extensions
# and never returns "audio", so audio must be matched by extension here.
# This set mirrors the frontend's audio detection (see
# ``getInlineFilePreviewKind`` in ``inline-file-preview-utils.ts``) — keep
# the two in sync.
_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".opus", ".flac", ".m4a", ".aac"}


def _is_inline_preview_media(filename: str) -> bool:
    """Whether ``filename`` is a video/audio artifact the frontend plays inline.

    These types are only detected by filename extension, since
    ``[label](file:id)`` targets are opaque UUIDs. The model is free to
    rewrite the label into descriptive prose that drops the extension (e.g.
    "下载视频（MP4）"), which would otherwise silently degrade the player to a
    plain download link — callers carry the real filename in the link
    *title* in that case (falling back to overwriting the label only when
    the filename can't safely become a title) so the extension survives
    into the rendered markdown either way. Deliberately limited to
    lightweight media players: office types (presentation/document/
    spreadsheet) keep the model's prose label because rewriting it would
    flip existing compact links into heavy inline preview boxes that
    eagerly fetch file bytes.
    """
    if artifact_type_for_filename(filename) == "video":
        return True
    return Path(filename).suffix.casefold() in _AUDIO_EXTENSIONS


# ``UploadedFile.filename`` has no charset constraint, so a user-uploaded
# file like ``clip ].mp4`` must never be substituted into a ``[label](...)``
# span unescaped: an embedded ``]`` (or ``[``/``\``) terminates the link
# early and the file reference is lost entirely — worse than the plain
# download link this rewrite is trying to improve on. Control characters
# are just as destructive through a different mechanism: a filename with a
# blank line splits the paragraph before the inline link is even parsed.
# Escaping instead of skipping would work for a single pass, but
# ``_MARKDOWN_FILE_REFERENCE_RE`` has no notion of backslash-escapes, so a
# later reconcile pass over already-escaped content would mis-locate the
# label boundary at the escaped bracket. Skipping the rewrite for these
# filenames is the safe, idempotent choice; the link still falls back to a
# plain download label.
#
# Used as the fallback path when a filename is unsafe for the *title*
# (below) — the two hazards are mostly disjoint, so a filename can safely
# carry a title while remaining unsafe as a label (or vice versa).
_UNSAFE_LABEL_RE = re.compile(r"[\[\]\\\x00-\x1f\x7f]")

# The primary mechanism (see ``reconcile_assistant_file_references``): carry
# the real filename in the link *title* rather than overwriting the label,
# so the model's own (possibly localized) prose survives in the persisted
# transcript. This must stay a superset of every character
# ``_MARKDOWN_FILE_REFERENCE_RE``'s title groups can't round-trip, so a
# title we emit here is always safely re-parsed by that regex on a later
# reconcile pass:
#   - ``"`` terminates the double-quote delimiter this function always
#     writes early — same class of hazard as ``]`` for a label.
#   - ``\`` (backslash) — same reasoning as ``_UNSAFE_LABEL_RE``: the regex
#     has no notion of CommonMark backslash-escapes, so raw stored text and
#     CommonMark-rendered text could otherwise diverge.
#   - ``)`` / ``[`` — the regex's title-parsing bounds a title to the
#     current link's own closing paren and never lets it cross into what
#     looks like another link's opening bracket; emitting either character
#     raw would defeat that on the very next reconcile pass.
#   - Control characters are unsafe for the same paragraph-splitting reason
#     as ``_UNSAFE_LABEL_RE``.
_UNSAFE_TITLE_RE = re.compile(r'["\\[)\x00-\x1f\x7f]')


def load_assistant_file_reference_records(
    db: Session,
    *,
    task_id: int,
    user_id: int,
) -> list[UploadedFile]:
    """Load the task/user file scope once for one or more reconciliations."""

    return (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == int(user_id),
            or_(UploadedFile.task_id == int(task_id), UploadedFile.task_id.is_(None)),
        )
        .all()
    )


def reconcile_assistant_file_references(
    db: Session,
    *,
    task_id: int,
    user_id: int,
    content: Any,
    records: Sequence[UploadedFile] | None = None,
) -> Any:
    """Canonicalize valid links, repair unique filename matches, and unlink fakes.

    Models occasionally copy a filename correctly while inventing the UUID in
    ``file:<id>``. A broken UUID must never become a clickable preview. When the
    markdown label uniquely identifies a file in the current task/user scope,
    replace it with that record's canonical id; otherwise keep only plain text.
    Filename repair is necessarily heuristic if an older same-named file is no
    longer present, so every repair is logged as a warning for auditability.
    """
    if not isinstance(content, str) or "file:" not in content:
        return content

    if records is None:
        records = load_assistant_file_reference_records(
            db,
            task_id=task_id,
            user_id=user_id,
        )
    records_by_id = {str(record.file_id): record for record in records}
    records_by_filename: dict[str, list[UploadedFile]] = defaultdict(list)
    task_records_by_filename: dict[str, list[UploadedFile]] = defaultdict(list)
    for record in records:
        filename_key = str(record.filename).casefold()
        records_by_filename[filename_key].append(record)
        if record.task_id is not None and int(record.task_id) == int(task_id):
            task_records_by_filename[filename_key].append(record)

    def replacement(match: re.Match[str]) -> str:
        prefix = match.group("image") or ""
        label = match.group("label")
        target = match.group("target")
        parsed_title = _parsed_title(match)
        referenced_id = parse_file_id_ref(target)
        record = records_by_id.get(referenced_id or "")

        if record is None:
            # The title is this function's own injection slot for the real
            # filename (see below) and takes priority over the label as the
            # repair source: once a reference has already been reconciled
            # once, a later pass over the same content (e.g. persist-time
            # then read-time) finds the filename in the title, not the
            # label, and must not lose repair capability just because of
            # that.
            filename = Path((parsed_title or label).strip()).name
            filename_key = filename.casefold()
            candidates = task_records_by_filename.get(filename_key, [])
            if not candidates:
                candidates = records_by_filename.get(filename_key, [])
            if len(candidates) == 1:
                record = candidates[0]
                logger.warning(
                    "Repaired invalid assistant FileRef %s using heuristic unique "
                    "filename %s for task %s",
                    referenced_id or target,
                    filename,
                    task_id,
                )

        if record is None:
            logger.warning(
                "Removed invalid assistant FileRef %s for task %s",
                referenced_id or target,
                task_id,
            )
            return label

        try:
            canonical_ref = build_file_id_ref(str(record.file_id))
        except ValueError:
            logger.warning(
                "Removed assistant FileRef %s for task %s because stored file id %s "
                "is invalid",
                referenced_id or target,
                task_id,
                record.file_id,
            )
            return label

        display_label = label
        display_title = parsed_title
        filename = str(record.filename or "")
        # Applies to ``![...]`` references too: _is_inline_preview_media only
        # matches video/audio, so genuine image alt text is never touched,
        # while a model that wraps a video in image syntax still gets a
        # title the frontend can classify (its image renderer resolves the
        # preview kind from title/alt and would otherwise fall back to a
        # broken img).
        if _is_inline_preview_media(filename):
            suffix = Path(filename).suffix.casefold()
            # Only intervene when the label alone can't already be
            # classified by the frontend -- mirrors the pre-title behavior
            # exactly, just preferring a title over overwriting the label
            # when intervention is needed. A label that already reveals the
            # type is left alone even if it names a different file than the
            # record (see test_reconcile_keeps_mismatched_label_...): there
            # is no signal that a coincidental extension match is wrong.
            label_reveals_type = bool(suffix) and label.strip().casefold().endswith(
                suffix
            )
            if not label_reveals_type:
                if not _UNSAFE_TITLE_RE.search(filename):
                    # Primary mechanism: carry the real filename in the
                    # title so the frontend can still detect the media
                    # type, while the model's own (possibly localized)
                    # label survives untouched in the persisted transcript.
                    # Always overwrites any parsed title so this stays
                    # idempotent and self-heals if the canonical filename
                    # ever changes.
                    display_title = filename
                elif not _UNSAFE_LABEL_RE.search(filename):
                    # Filename has a quote character and can't safely
                    # become a title. Fall back to the pre-title mechanism:
                    # overwrite the label itself. Drop any parsed title
                    # too, so a stale title never survives alongside a
                    # freshly rewritten label.
                    display_title = None
                    display_label = filename
                # else: unsafe for both title and label -- leave the
                # reference untouched; it degrades to a plain download
                # link on the frontend, same as any other undetectable
                # file type.

        # A pass-through title (parsed from single-quote syntax, which
        # allows an embedded double quote) could be unsafe to re-emit with
        # the double-quote delimiter this function always writes. Drop it
        # rather than escape it -- same reasoning as `_UNSAFE_LABEL_RE`.
        if display_title and not _UNSAFE_TITLE_RE.search(display_title):
            return f'{prefix}[{display_label}]({canonical_ref} "{display_title}")'
        return f"{prefix}[{display_label}]({canonical_ref})"

    return _MARKDOWN_FILE_REFERENCE_RE.sub(replacement, content)
