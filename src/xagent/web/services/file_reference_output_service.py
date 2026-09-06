"""Validate and repair model-authored file links before they reach the UI."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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
# The whitespace immediately around a title clause is deliberately ``[ \t]``
# (horizontal only), not ``\s`` -- ``\s`` includes newlines, which would let
# a title clause span a CommonMark blank line, e.g.
# ``[a](file:id\n\n"stale.mp4")`` renders as two independent, inert literal
# paragraphs to any real CommonMark parser, but ``\s+``/``\s*`` here would
# greedily match across that blank line and emit one live link, silently
# merging (and mis-rendering) two paragraphs. Known cost: CommonMark also
# permits a *single* line ending between destination and title (still one
# link when rendered), and horizontal-only whitespace declines to match
# that shape too -- such a reference is left byte-for-byte untouched (see
# test_reconcile_leaves_single_newline_titled_reference_untouched), the
# same non-destructive fallback as any other unparsable input, rather than
# validated. Accepted: this function only ever emits a single space there,
# so the shape is model-authored and rare, and distinguishing one newline
# from two would complicate exactly the machinery this pass is shrinking.
#
# A title clause that doesn't parse as any of the three forms (duplicated,
# unterminated, mismatched delimiters) falls through to the named ``junk``
# alternative, which discards it as unstructured trailing content up to the
# next ``)`` -- the reference is still validated, just without a title. A
# deliberate side effect: an input like ``[x](file:id not a title)``, which
# CommonMark renders as inert literal text (invalid destination/title
# syntax), is normalized into a live ``[x](file:id)`` link when the id
# resolves. That's this function's purpose applied to a slightly mangled
# reference -- the model clearly meant a file link -- rather than an
# accident; see test_reconcile_normalizes_unparsable_trailing_junk. That
# junk alternative excludes ``(`` (as well as ``)``/``[``/control chars): if
# it didn't, junk could cross an inner ``)`` that isn't this link's own
# closing paren and strand the remainder of the message outside the link
# entirely (verified: ``[x](file:id (c) 2024) tail`` would otherwise match
# only through the first ``)``, leaving `` 2024) tail`` as stray text in the
# persisted transcript). Excluding ``(`` means this and similar malformed
# inputs simply fail to match at all -- identical to this regex's pre-title
# behavior of leaving an unparsable reference completely untouched, which
# is the correct, non-destructive fallback. But matching non-trivial junk is
# not by itself a guarantee the reference is safe to normalize: if the id
# turns out not to resolve to any record, ``replacement()`` below returns
# the entire original match unchanged rather than collapsing to the bare
# label -- otherwise the junk text (ordinary trailing prose the model wrote,
# not link syntax) would be permanently deleted from the persisted
# transcript alongside the invalid reference.
#
# Each title form also tolerates trailing whitespace before the closing
# ``)`` (``[ \t]*`` after the delimiter closes) -- otherwise
# ``[a](file:id "t.mp4"  )`` would fail title-form and fall through to junk,
# which has no notion of title syntax and would silently discard the title
# text with no re-injection for non-media files.
#
# ``target`` and ``label`` are each wrapped in an atomic group
# (``(?>...)``, Python 3.11+). Both character classes already stop at the
# first disallowed character, so a successful match never needs to
# backtrack into either group -- the only case that would is an unmatchable
# input (e.g. no ``)`` anywhere in the whole remaining string), where
# ``target``'s ``+`` and the ``junk`` alternative's ``*`` can otherwise trade
# an overlapping character back and forth one at a time, producing quadratic
# blowup on attacker-influenceable content this function re-runs on every
# read (measured: ~3.5s to fail a single ~32KB unclosed reference without
# the atomic group; sub-millisecond with it, at 6x that size). The atomic
# group cannot change any successful match's result, only how fast an
# unmatchable one fails.
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
# ``iter_markdown_file_references`` /
# ``replace_markdown_file_references`` below are that shared helper: a
# consumer needing to read or rewrite this format goes through them, so
# there is one pattern to keep correct rather than one per consumer.
_MARKDOWN_FILE_REFERENCE_RE = re.compile(
    r"(?P<image>!)?\[(?P<label>(?>[^\]]*))\]\("
    r"(?P<target>(?>file:[^)\s]+))"
    r"(?:"
    r"(?:[ \t]+"
    r'(?:"(?P<dtitle>(?:[^"\\)[\x00-\x1f\x7f]|\\.)*)"'
    r"|'(?P<stitle>(?:[^'\\)[\x00-\x1f\x7f]|\\.)*)'"
    r"|\((?P<ptitle>(?:[^()\\[\x00-\x1f\x7f]|\\.)*)\))"
    r"[ \t]*)?"
    r"|(?P<junk>[^()[\x00-\x1f\x7f]*)"
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


@dataclass(frozen=True)
class MarkdownFileReference:
    """One parsed ``[label](file:id ["title"])`` reference.

    The public, structured view of a single ``_MARKDOWN_FILE_REFERENCE_RE``
    match, so a consumer outside this module can read a reference's parts
    without owning a second copy of the grammar. ``file_id`` is already
    through ``parse_file_id_ref``, so a real ``file:///path`` URI never
    reaches a consumer as a handle.

    ``span`` indexes the *original* string, which is what makes a
    replacement-by-span rewrite possible; ``text`` is the matched substring,
    so a consumer that decides to leave a reference alone can put back
    exactly what was written.

    **Caller-facing contract details that are easy to get wrong:**

    * ``label`` can be the empty string -- ``[](file:id)`` is a well-formed
      match. Every existing consumer in this repo substitutes something
      (``label or "image"``, ``label or "file"``); a caller that renders it
      bare gets an empty link text.
    * ``file_id`` is percent-*decoded* (``parse_file_id_ref`` runs
      ``unquote``), so it is not necessarily a literal substring of
      ``content``: ``[a](file:a%20b)`` yields ``file_id="a b"``. A caller
      re-emitting a target MUST go through ``build_file_id_ref`` to
      re-encode it -- ``f"file:{reference.file_id}"`` can produce a target
      that no longer matches this module's own regex.
    * ``title`` is raw: backslash escapes inside it are not decoded.
    * ``junk`` is the reference's trailing unparsable prose -- see below.
      A caller that rewrites a reference without consulting it will delete
      that prose from the output.
    """

    file_id: str
    label: str
    title: str | None
    is_image: bool
    span: tuple[int, int]
    text: str
    # Trailing content inside the parens that parsed as neither destination
    # nor title -- ``[the docs](file:id for more details)`` yields
    # ``" for more details"``.
    #
    # Exposed because a caller cannot otherwise tell this case apart from an
    # ordinary untitled reference (``title`` is ``None`` either way), and the
    # difference decides whether rewriting is lossless. The regex matched
    # that text as link syntax, but it is ordinary prose the model wrote:
    # replacing the whole match discards it permanently. That is the
    # destructive-loss class issue #1202 asked this module to eliminate, and
    # ``reconcile_assistant_file_references`` guards it by falling back to
    # ``match.group(0)`` whenever a reference carrying non-blank junk turns
    # out to be unrecoverable.
    #
    # This function cannot make that call itself: whether a reference is
    # recoverable depends on the caller's own lookup (the sibling has the
    # database; this has a callback). So the decision is handed over rather
    # than guessed -- see ``replace_markdown_file_references``, which
    # defaults to preserving.
    junk: str = ""


def iter_markdown_file_references(content: str) -> Iterator[MarkdownFileReference]:
    """Every internal ``file:`` reference in ``content``, in order.

    The reading half of this module's canonical parser. The comment above
    ``_MARKDOWN_FILE_REFERENCE_RE`` asks a consumer that needs to match this
    format to route through a shared helper rather than fork the pattern;
    until now there was no such helper, so the only options were forking it
    or importing a private constant. A fork drifts silently -- a title-blind
    copy reads a title clause as part of the id, and a copy without the
    ``junk`` alternative fails to match shapes this one recovers -- and the
    drift only shows up as a mangled reference in somebody's transcript.

    See ``MarkdownFileReference`` for the contract details a caller has to
    handle: an empty ``label``, a percent-decoded ``file_id``, and ``junk``.

    Targets that are not internal handles are skipped rather than yielded
    with an empty id: whether a ``file:`` target is a handle at all is
    ``parse_file_id_ref``'s decision, and a consumer should not have to
    re-make it.

    Content with no ``file:`` substring is skipped without running the regex.
    That is correctness-preserving (every match contains the literal), and it
    is here for cost: the pattern's atomic label group makes ``finditer``
    explore bracket-run candidates, and ``"[" * 100000 + "]"`` measures at
    ~1.5s without the precheck and is instant with it.

    It is a common-case optimisation, **not** a defence against adversarial
    input: one ``file:`` anywhere in the content passes the precheck and the
    same pathological cost returns (``"file:" + "[" * 100000 + "]"`` measures
    ~2s either way). Hardening that would mean changing the pattern, which
    this addition deliberately does not touch.

    Deliberately *not* guarded on ``isinstance(content, str)``, unlike
    ``reconcile_assistant_file_references``: a non-string must still raise --
    once iterated. This is a generator, so no body runs until the first
    ``next()``; an unconsumed ``iter_markdown_file_references(None)`` raises
    nothing. Returning early instead would turn bad input into an empty
    iteration a caller cannot tell apart from "no references".

    The precheck is therefore ordered *after* a ``str``-only operation, not
    before one. ``"file:" not in content`` is true for a ``list``/``tuple``/
    ``set``/``dict`` of strings as readily as for text, so testing it first
    would swallow exactly the bad input this contract promises to reject --
    ``finditer`` used to raise on those and must keep doing so.
    """
    if not isinstance(content, str):
        raise TypeError(f"content must be a str, got {type(content).__name__}")
    if "file:" not in content:
        return
    for match in _MARKDOWN_FILE_REFERENCE_RE.finditer(content):
        reference = _reference_from_match(match)
        if reference is not None:
            yield reference


def _reference_from_match(match: re.Match[str]) -> MarkdownFileReference | None:
    """One match as a reference, or ``None`` if its target is not a handle.

    Shared by both public entry points so they cannot disagree about what a
    reference is, or about which matches are skipped.
    """
    file_id = parse_file_id_ref(match.group("target"))
    if not file_id:
        return None
    return MarkdownFileReference(
        file_id=file_id,
        label=match.group("label"),
        title=_parsed_title(match),
        is_image=bool(match.group("image")),
        span=match.span(),
        text=match.group(0),
        junk=match.group("junk") or "",
    )


def replace_markdown_file_references(
    content: str,
    replace: Callable[[MarkdownFileReference], str],
    *,
    rewrite_junk_references: bool = False,
) -> str:
    """Rewrite every internal ``file:`` reference through ``replace``.

    The writing half, for a consumer whose job is to *remove* the ``file:``
    scheme rather than canonicalize it -- a chat channel whose client cannot
    resolve the handle, say. ``reconcile_assistant_file_references`` does not
    serve that consumer: it is the function that puts canonical ``file:``
    links *into* content.

    Non-handle targets are left exactly as written, for the same reason they
    are skipped by ``iter_markdown_file_references``.

    **A reference carrying non-blank junk is left untouched by default.**
    ``[the docs](file:id for more details)`` matches because the regex's
    ``junk`` alternative accepts trailing unparsable content, but that
    content is ordinary prose the model wrote, not link syntax -- rewriting
    the whole match deletes a fragment of the user's sentence. That is the
    destructive-content-loss class issue #1202 closed on the reconcile path,
    and defaulting to "rewrite" here would reopen it for every consumer that
    did not know to ask.

    ``reconcile_assistant_file_references`` makes this call conditionally:
    junk is discardable when the id *resolves* to a record, and preserved
    when the reference is unrecoverable. This function cannot decide that
    itself -- recoverability lives in the caller's lookup, not here -- so the
    conservative half is the default and ``rewrite_junk_references=True``
    opts into the other. A caller that opts in should consult
    ``MarkdownFileReference.junk`` and re-emit the prose it is about to
    consume; returning ``reference.text`` from ``replace`` is the equivalent
    of leaving one such reference alone case by case.

    Substitution runs through ``re.sub``, so the output is built in one pass
    and a replacement's length cannot disturb a later reference's position.
    ``re.sub`` calls the callback in *forward* order, once per match.

    ``replace`` must return a ``str``. A callback that returns ``None`` -- an
    accidentally missing ``return`` is the realistic way this happens --
    raises ``TypeError`` here rather than being honoured: ``re.sub`` treats a
    ``None`` replacement as the empty string and would *silently delete*
    every reference it was handed, which is the same destructive-content-loss
    outcome the junk guard above exists to prevent. A callback that *raises*
    is left alone: the exception propagates and ``content`` is unmutated,
    because ``re.sub`` builds the result separately.
    """

    def _substitute(match: re.Match[str]) -> str:
        reference = _reference_from_match(match)
        if reference is None:
            # Not an internal handle. Put back exactly what was written --
            # deciding that is ``parse_file_id_ref``'s job, not this one's.
            return match.group(0)
        if reference.junk.strip() and not rewrite_junk_references:
            return match.group(0)
        replacement = replace(reference)
        if not isinstance(replacement, str):
            raise TypeError(
                "replace must return a str; got "
                f"{type(replacement).__name__} for file reference "
                f"{reference.file_id!r}"
            )
        return replacement

    if not isinstance(content, str):
        # See ``iter_markdown_file_references``: the type rejection has to
        # precede the containment test, which a container of strings passes.
        raise TypeError(f"content must be a str, got {type(content).__name__}")
    if "file:" not in content:
        # Same short-circuit and the same reasoning as
        # ``iter_markdown_file_references``. Returning ``content`` is exact:
        # with no ``file:`` there is no match, so ``sub`` would return it
        # unchanged anyway.
        return content
    return _MARKDOWN_FILE_REFERENCE_RE.sub(_substitute, content)


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
#
# The ``)``/``[`` exclusions are an implementation constraint of this
# module's own re-parse requirement, not a CommonMark rule -- CommonMark
# itself permits an unescaped ``)`` inside a double-quoted title. A real
# filename containing one, e.g. ``video (1).mp4``, is therefore denied a
# title here and falls back to the pre-title mechanism instead (overwriting
# the label directly, see below) -- media type detection still works, just
# via the older, more destructive path. Escaping ``)``/``[`` on emission
# (the regex's ``\\.`` alternative already accepts an escaped one) would
# close this gap, but titles round-trip through several call sites that
# compare the raw parsed value against the filename (self-healing, the
# repair heuristic above) -- escaping would need a matching unescape step
# at every one of them to stay correct, which is a larger, separate change
# from the narrow fixes in this pass. See
# test_reconcile_falls_back_to_label_rewrite_for_paren_in_filename.
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
            UploadedFile.detached_reason.is_(None),
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
    ``file:<id>``. A broken UUID must never become a clickable preview. Repair
    tries the markdown *title* before the label -- once a reference has
    already been through one reconcile pass, the real filename lives there,
    not in the label -- falling through to the label if the title doesn't
    uniquely identify a file in the current task/user scope. If neither does,
    keep only plain text. Filename repair is necessarily heuristic if an
    older same-named file is no longer present, so every repair is logged as
    a warning for auditability.
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
        # filename is NOT NULL in the schema, but skip defensively rather
        # than index a None record under the literal string "none" (via
        # str(None).casefold()) -- a title/label candidate of literally
        # "None" is unlikely but not impossible, and there is no reason for
        # this index to ever produce a false match for it.
        if record.filename is None:
            continue
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

        # Non-whitespace junk (see the "junk" alternative's comment above)
        # was ordinary trailing prose the model wrote, not link syntax the
        # regex is entitled to discard. If the reference turns out to be
        # unrecoverable below, giving up must mean "leave the original text
        # exactly as written" -- never "keep the junk's surrounding syntax
        # but delete the junk itself".
        junk = match.group("junk")
        if junk and junk.strip():
            unrecoverable_fallback = match.group(0)
        elif (
            parsed_title
            and parsed_title.strip()
            and parsed_title.strip() != label.strip()
        ):
            # The title is this function's own designated channel for the
            # real filename (see below) -- on give-up, dropping it bare
            # would destroy the one piece of information naming which file
            # was meant, while the label may be generic model prose with no
            # filename in it at all. That recreates, in this function's own
            # new code, exactly the destructive-loss failure class #1202
            # asked this function to eliminate. Preserving it as plain text
            # alongside the label costs nothing: the reference is already
            # being unlinked, so there is no markdown structure left to
            # keep it safe for. Guarded against a whitespace-only title
            # (nothing worth preserving) and a title identical to the label
            # (would render a pointless "x (x)") -- both fall through to
            # the plain label case below instead.
            unrecoverable_fallback = (
                f"{label} ({parsed_title})" if label else parsed_title
            )
        else:
            unrecoverable_fallback = label

        if record is None:
            # Try the title before the label, but fall through to the label
            # if the title doesn't resolve to exactly one record -- not just
            # if it's absent. The title is this function's own injection
            # slot for the real filename (see below), so once a reference
            # has already been reconciled once, a later pass over the same
            # content (e.g. persist-time then read-time) finds the filename
            # there rather than in the label, and must not lose repair
            # capability just because of that. But an unconditional
            # title-over-label pick would create the mirror-image gap: a
            # model-authored title that happens to name a different (or no)
            # file must not shadow a label that's otherwise a perfect,
            # resolvable match.
            for filename_source in (parsed_title, label):
                if not filename_source:
                    continue
                filename = Path(filename_source.strip()).name
                if not filename:
                    # filename_source was whitespace-only (or otherwise
                    # stripped to nothing) -- skip it rather than looking up
                    # the empty-string key, which could otherwise match a
                    # record whose own filename is empty/whitespace-only.
                    continue
                filename_key = filename.casefold()
                candidates = task_records_by_filename.get(filename_key, [])
                if not candidates:
                    candidates = records_by_filename.get(filename_key, [])
                if len(candidates) == 1:
                    record = candidates[0]
                    logger.warning(
                        "Repaired invalid assistant FileRef %s using heuristic "
                        "unique filename %s for task %s",
                        referenced_id or target,
                        filename,
                        task_id,
                    )
                    break

        if record is None:
            logger.warning(
                "Removed invalid assistant FileRef %s for task %s",
                referenced_id or target,
                task_id,
            )
            return unrecoverable_fallback

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
            return unrecoverable_fallback

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
            # classified by the frontend -- the same gate the pre-title
            # label-rewrite used (the mechanism differs: inject/overwrite a
            # title first, only falling back to overwriting the label when
            # the filename can't safely become a title). A label that
            # already reveals the type is left alone even if it names a
            # different file than the record (see
            # test_reconcile_keeps_mismatched_label_...): there is no
            # signal that a coincidental extension match is wrong.
            # _is_inline_preview_media(filename) above guarantees suffix is
            # non-empty, so no separate bool(suffix) check is needed here.
            label_reveals_type = label.strip().casefold().endswith(suffix)
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
        #
        # This delimiter choice also means a model-authored '...' or (...)
        # title is normalized to "..." syntax even when nothing else about
        # the reference changed -- intentional (one consistent output form
        # is simpler to reason about than round-tripping the original
        # delimiter), not an oversight.
        if display_title and not _UNSAFE_TITLE_RE.search(display_title):
            return f'{prefix}[{display_label}]({canonical_ref} "{display_title}")'
        return f"{prefix}[{display_label}]({canonical_ref})"

    return _MARKDOWN_FILE_REFERENCE_RE.sub(replacement, content)
