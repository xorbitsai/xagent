"""Composio's own Drive tools for the three byte-writing Drive tools.

The proxy transport carries JSON calls and relays a binary answer, but it
cannot carry a multipart or resumable upload: a Drive create or content
update sends metadata and bytes in one request. On a Composio-backed grant
google_drive.py therefore stages the bytes through Composio's Files API
(``composio_tools``, the toolkit-agnostic transport) and runs Composio's
Drive tool on the staged object. This module holds that write path and the
verification that follows it.

A tool's answer is never taken for the result: a create is read back through
the proxy, so the checksum and the type are Drive's own. An answer that was
lost keeps the outcome unknown -- a same-name file is reported to the caller
as a candidate to check, never as this call's result, because Drive allows
duplicate names and nothing the lookup can see identifies the operation.

This module reads the content-update primitives (messages, metadata
summaries, checksum comparison) that google_drive.py owns for its native
path. google_drive.py therefore imports this module on a Composio branch
only, not at its module import, which keeps the two acyclic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, BinaryIO, Literal

from . import composio_tools
from .google_drive import (
    _DRIVE_ID_CHARS,
    _HASH_BLOCK_BYTES,
    _LOCAL_FILE_CHANGED,
    _NOTHING_CHANGED,
    _OCTET_STREAM,
    _as_int,
    _block_digest,
    _content_error_detail,
    _content_permission_message,
    _content_summary,
    _content_update_error,
    _content_update_mismatch,
    _content_update_success,
    _file_changed_since,
    _file_mtime_ns,
    _hash_replacement_file,
    _is_google_workspace_mime_type,
    _normalize_mime_type,
    _read_content_update_file,
    _read_exactly,
    _ReplacementChangedError,
    _same_content,
    _then_nothing_changed,
    _unavailable_content_update_message,
)
from .utils import (
    google_api_error_status,
    is_google_file_access_error,
    is_google_file_unavailable_error,
)

logger = logging.getLogger("google-drive-mcp")

# The slug the Drive toolkit is registered under in Composio. It names the
# toolkit for the Files API; the display name below is only for a message a
# caller reads.
GOOGLEDRIVE_TOOLKIT_SLUG = "googledrive"
GOOGLEDRIVE_TOOLKIT_NAME = "Google Drive"
# The two Drive tools that move bytes. A create carries metadata and content
# in one request; an update is a media-only upload. Neither fits Proxy
# Execute, which takes one JSON body and no multipart or resumable session.
GOOGLEDRIVE_CREATE_FILE = "GOOGLEDRIVE_CREATE_FILE"
GOOGLEDRIVE_UPLOAD_UPDATE_FILE = "GOOGLEDRIVE_UPLOAD_UPDATE_FILE"

# The file fields both create paths return, matching the native requests.
_CREATED_FILE_FIELDS = "id, name, webViewLink, mimeType"
# The same read plus what a post-create integrity check needs. The extra
# fields are used for the check and not added to the result, whose shape
# stays the four fields above, as native.
_CREATED_FILE_READ_FIELDS = f"{_CREATED_FILE_FIELDS}, md5Checksum, size"
# What a lost create's same-name lookup reports for a caller to check by
# hand: nothing here is evidence that this call created the file.
_CREATED_FILE_CANDIDATE_FIELDS = "id, name, createdTime"

# A staged upload keeps a file at most this large in memory, so a small
# upload reads it once and a larger one streams it with block checks.
_STAGED_UPLOAD_IN_MEMORY_BYTES = 5_000_000

# What a failed update call established: "definite" and "unclear" are the two
# ways Composio itself can fail it, "reported_failure" is a tool that ran and
# answered that it failed (see _composio_update_failure).
_Refusal = Literal["definite", "unclear", "reported_failure"]


# The values Composio's GOOGLEDRIVE_CREATE_FILE documents for ``mimeType``
# (Composio's Drive toolkit documentation, toolkit version 20261104_00,
# fetched 2026-11-10). It is a closed enum, while a native create accepts
# any type: mimetypes.guess_type and google_drive.py's _TEXT_MIME_TYPES
# produce values outside it (application/msword, application/vnd.ms-excel,
# text/x-python, application/javascript, text/css, ...). Sending one of
# those would turn an upload that works on a native grant into a definite
# rejection, so _create_tool_mime_type() sends application/octet-stream
# instead. Whether Composio enforces the enum is a stage check (listed in
# the PR's stage checks); if it turns out not to, this mapping can go.
_CREATE_TOOL_MIME_TYPES = frozenset(
    {
        "application/vnd.google-apps.folder",
        "application/vnd.google-apps.document",
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.google-apps.presentation",
        "application/vnd.google-apps.drawing",
        "application/vnd.google-apps.form",
        "application/vnd.google-apps.script",
        "application/vnd.google-apps.shortcut",
        "application/vnd.google-apps.site",
        "application/vnd.google-apps.map",
        "application/vnd.google-apps.jam",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.oasis.opendocument.text",
        "application/rtf",
        "text/plain",
        "text/html",
        "text/markdown",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.oasis.opendocument.spreadsheet",
        "text/csv",
        "text/tab-separated-values",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.oasis.opendocument.presentation",
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/webp",
        "image/svg+xml",
        "image/tiff",
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-ms-wmv",
        "video/webm",
        "audio/mpeg",
        "audio/wav",
        "audio/ogg",
        "application/zip",
        "application/vnd.rar",
        "application/x-tar",
        "application/gzip",
        "application/x-7z-compressed",
        "application/json",
        "application/xml",
        "application/x-yaml",
        "application/epub+zip",
        _OCTET_STREAM,
    }
)


def _holds_content(file: dict[str, Any], *, md5: str, size: int) -> bool:
    """Whether ``file`` is Drive's record of exactly the bytes uploaded."""
    stored_md5 = file.get("md5Checksum")
    if not isinstance(stored_md5, str) or not stored_md5:
        return False
    if stored_md5.lower() != md5.lower():
        return False
    stored_size = _as_int(file.get("size"))
    return stored_size is None or stored_size == size


class _VerifiedStagedReader:
    """The hashed bytes of ``fh``, checked block by block while they upload.

    A staged upload reads the replacement file again after the hash pass.
    Each block is read at most once, checked against its digest, and then
    served from a one-block cache, so a small read (``requests`` asks for
    16 KiB at a time) does not re-read the whole 1 MiB block. A block is
    never served before its digest matches, so the bytes that are sent are
    the bytes that were hashed. If the file changes after a block was
    checked, the cached copy keeps being sent and the next block stops the
    upload. Only a temporary object is being written, so Drive is never
    touched by a stopped upload.
    """

    def __init__(self, fh: BinaryIO, *, size: int, block_digests: list[bytes]) -> None:
        self._fh = fh
        self._size = size
        self._block_digests = block_digests
        self._offset = 0
        self._block_index = -1
        self._block = b""

    def __len__(self) -> int:
        # What requests uses for Content-Length, so the body is not chunked.
        return self._size

    def _read_block(self, index: int) -> None:
        start = index * _HASH_BLOCK_BYTES
        stop = min(start + _HASH_BLOCK_BYTES, self._size)
        self._fh.seek(start)
        block = _read_exactly(self._fh, stop - start)
        if _block_digest(block) != self._block_digests[index]:
            raise _ReplacementChangedError(_LOCAL_FILE_CHANGED)
        self._block_index = index
        self._block = block

    def read(self, amount: int = -1) -> bytes:
        end = (
            self._size
            if amount is None or amount < 0
            else min(self._offset + amount, self._size)
        )
        if self._offset >= end:
            return b""
        parts: list[bytes] = []
        while self._offset < end:
            index = self._offset // _HASH_BLOCK_BYTES
            if index != self._block_index:
                self._read_block(index)
            block_start = index * _HASH_BLOCK_BYTES
            take = min(end, block_start + len(self._block)) - self._offset
            beginning = self._offset - block_start
            parts.append(self._block[beginning : beginning + take])
            self._offset += take
        return b"".join(parts)


def _create_tool_mime_type(mime_type: str) -> str:
    """``mime_type`` as ``GOOGLEDRIVE_CREATE_FILE`` accepts it.

    The tool's ``mimeType`` is a closed enum (_CREATE_TOOL_MIME_TYPES), so a
    value outside it is sent as ``application/octet-stream``: the file is
    still created with the right bytes, only its label falls back. A Google
    Docs/Sheets/Slides conversion target is one of the enum's own values and
    is kept, so the conversion is unaffected. The staged object's ``mimetype``
    keeps the real type either way.
    """
    normalized = _normalize_mime_type(mime_type)
    if normalized in _CREATE_TOOL_MIME_TYPES:
        return normalized
    return _OCTET_STREAM


def _needs_exact_create_type(target_mime_type: str | None) -> bool:
    """Whether a created file's type has to equal ``target_mime_type``.

    Only a Google Docs/Sheets/Slides conversion changes the file's type
    inside the create, so only that kind of target is compared exactly: a
    file that came back as another type was not converted, and the caller
    asked for the converted file. For any other target the tool may send the
    enum's fallback (see ``_create_tool_mime_type``) while the staged object
    keeps the real type, and which of the two labels Drive stores is not
    settled, so the bytes are the evidence and a different label is not a
    failed create. A caller that can never convert passes ``None``.
    """
    return target_mime_type is not None and _is_google_workspace_mime_type(
        target_mime_type
    )


def _tool_file_object(data: Any) -> dict[str, Any]:
    """The file object in a Drive tool's answer, or an empty mapping.

    Composio does not document the create tool's output shape. A nested
    ``file``/``data`` object is preferred over the top level: a top-level
    ``id`` can be the tool execution's own id rather than the new file's.
    Only an id with Drive's own id shape is accepted, so a blank or
    execution-shaped value is not reported as the new file.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return {}
    if not isinstance(data, dict):
        return {}
    candidates: list[dict[str, Any]] = []
    for key in ("file", "data"):
        nested = data.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
    candidates.append(data)
    for candidate in candidates:
        file_id = candidate.get("id")
        if isinstance(file_id, str) and _DRIVE_ID_CHARS.fullmatch(file_id):
            return candidate
    return {}


def _created_file_metadata(
    service: Any, file_id: str, tool_data: Any
) -> tuple[dict[str, Any] | None, bool]:
    """The new file's metadata, matching what the native create path returns,
    and whether Drive itself was read for it.

    Read back through the proxy so the output shape does not depend on the
    Composio tool's answer. The extra checksum and size fields are used by
    the caller's integrity check and dropped from the result. If the read
    fails, the tool's answer (which at least names the new id) is used
    instead of reporting a failure: the file exists, and a retry would
    create a second one. That fallback can carry as little as the id, and
    whatever type it carries is the tool's claim, not what Drive stored, so
    the second element is what tells a caller which of the two it has. A
    read that answers 404 is different: Drive says no file has that id, so
    the id proves nothing and ``(None, False)`` is returned for the caller to
    report the same-name files it finds instead.
    """
    try:
        file = (
            service.files()
            .get(
                fileId=file_id,
                supportsAllDrives=True,
                fields=_CREATED_FILE_READ_FIELDS,
            )
            .execute()
        )
        if isinstance(file, dict):
            return file, True
    except Exception as exc:
        logger.warning("Could not read the created file %s: %s", file_id, exc)
        if google_api_error_status(exc) == 404:
            return None, False
    fallback = {
        key: value
        for key, value in _tool_file_object(tool_data).items()
        if key in ("id", "name", "webViewLink", "mimeType") and isinstance(value, str)
    }
    fallback.setdefault("id", file_id)
    return fallback, False


def _created_file_output(file: dict[str, Any]) -> dict[str, Any]:
    """The four fields both native create paths return."""
    return {
        key: file[key]
        for key in ("id", "name", "webViewLink", "mimeType")
        if key in file
    }


def _created_content_mismatch(
    file: dict[str, Any], *, md5: str, size: int
) -> str | None:
    """Why Drive's stored file is not the bytes that were uploaded, or None.

    A Google Workspace file has no MD5 (its bytes are converted), so it is
    not checked. A stored file with neither a checksum nor a size cannot be
    checked either; that gap is covered by the post-upload read-back being
    the only evidence available.
    """
    stored_md5 = file.get("md5Checksum")
    if isinstance(stored_md5, str) and stored_md5:
        if stored_md5.lower() != md5.lower():
            return "Google Drive's checksum does not match the bytes that were uploaded"
        stored_size = _as_int(file.get("size"))
        if stored_size is not None and stored_size != size:
            return f"Google Drive reports {stored_size} bytes, not {size}"
        return None
    if _is_google_workspace_mime_type(str(file.get("mimeType") or "")):
        return None
    stored_size = _as_int(file.get("size"))
    if stored_size is not None and stored_size != size:
        return f"Google Drive reports {stored_size} bytes, not {size}"
    return None


def _drive_query_literal(value: str) -> str:
    """``value`` as a single-quoted Drive query string literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _create_candidate(file: dict[str, Any]) -> dict[str, str]:
    """A same-name file as reported to the caller: what to check, not proof.

    Only the fields a person needs to find the file in Drive: the id, the
    name and the creation time. A checksum, size or type is not reported
    because none of them identifies this call's own file either.
    """
    entry = {"id": str(file.get("id") or "")}
    for key in ("name", "createdTime"):
        value = file.get(key)
        if isinstance(value, str) and value:
            entry[key] = value
    return entry


def _create_candidate_label(candidate: dict[str, str]) -> str:
    """One candidate in a sentence: ``id 'doc1' (created <time>)``."""
    label = f"id '{candidate['id']}'"
    if "createdTime" in candidate:
        label += f" (created {candidate['createdTime']})"
    return label


def _same_name_files(
    service: Any, *, name: str, parent_id: str | None
) -> tuple[list[dict[str, Any]], Exception | None]:
    """Files Drive already has under this name and parent.

    A create whose answer was lost is looked up by name and parent so the
    caller can be told what Drive holds. These files are only diagnostic:
    Drive allows duplicate names, and neither the tool's request nor the
    create itself leaves an operation-specific marker on the file, so
    nothing collected here -- name, parent, creation time, type or checksum,
    or all of them together -- shows that this call created the file. A
    file made earlier with the same name, folder and bytes matches every one
    of those checks. The caller therefore keeps the outcome unknown and
    lists these files for the user to check; it never reports one of them as
    this create's result.

    Shared drives are searched too: without ``corpora="allDrives"`` a file
    created in one is never even named as a candidate.
    """
    query = f"name = '{_drive_query_literal(name)}' and trashed = false"
    if parent_id is None:
        query += " and 'root' in parents"
    else:
        query += f" and '{_drive_query_literal(parent_id)}' in parents"
    try:
        answer = (
            service.files()
            .list(
                q=query,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                # Without this a file in a shared drive is never found.
                corpora="allDrives",
                pageSize=10,
                # The newest first: the closest thing to the file this call
                # may have made, for a reader to check first.
                orderBy="createdTime desc",
                fields=f"files({_CREATED_FILE_CANDIDATE_FIELDS})",
            )
            .execute()
        )
    except Exception as exc:
        logger.warning("Could not look for the new file %r: %s", name, exc)
        return [], exc
    files = answer.get("files") if isinstance(answer, dict) else None
    if not isinstance(files, list):
        return [], RuntimeError("Drive returned an invalid file list")
    return [
        file
        for file in files
        if isinstance(file, dict) and isinstance(file.get("id"), str) and file["id"]
    ], None


def _unconfirmed_create_message(
    *,
    action: str,
    name: str,
    candidates: list[dict[str, Any]],
    lookup_error: Exception | None,
    reason: str | None = None,
) -> str:
    """What to tell a caller when a create tool's answer was lost.

    ``action`` is "uploaded" or "created". The wording never claims a
    definite failure while the file may exist, and never invites a blind
    retry that would make a duplicate. In particular, a lookup that finds
    nothing is not proof: the tool may still be running in Composio, and a
    write can land after this read, so the result of this call stays
    unknown. Nor is a same-name file proof: it is reported as a candidate to
    check, because it may be this call's file or an earlier one. ``reason``
    is what the tool itself said when it reported a failure; it does not
    settle the outcome, but it can tell the user what to check.
    """
    if lookup_error is not None:
        message = (
            f"Composio did not confirm that the file was {action}, and Google "
            "Drive could not be checked for it, so it is not known whether it "
            f"exists. Do not retry blindly: check Google Drive for '{name}' "
            "first; a retry could create a second file."
        )
    elif candidates:
        shown = "; ".join(
            _create_candidate_label(_create_candidate(candidate))
            for candidate in candidates[:3]
        )
        if len(candidates) > 3:
            shown += f"; and {len(candidates) - 3} more"
        found = f"{len(candidates)} files" if len(candidates) > 1 else "a file"
        message = (
            f"Composio did not confirm that the file was {action}, and Google "
            f"Drive already has {found} named '{name}' in that folder that "
            "cannot be shown to be this call's result: "
            f"{shown}. It may be the file this call was making, or an earlier "
            "one; a name, folder and creation time are not proof of which. "
            f"It is not known whether this call {action} a file. Check those "
            "files in Google Drive before retrying; a retry could create a "
            "duplicate."
        )
    else:
        noun = "upload" if action == "uploaded" else "create"
        message = (
            f"Composio did not confirm that the file was {action}. A check of "
            f"Google Drive found no file named '{name}' in that folder yet; the "
            f"{noun} may still be finishing in Composio, so it is not known "
            f"whether it was {action}. Check Google Drive again before "
            "retrying; a retry could create a duplicate."
        )
    if reason:
        message = f"{message} Composio reported: {reason}"
    return message


def _composio_rejection_is_definite(exc: composio_tools.ComposioToolError) -> bool:
    """Whether Composio's refusal proves the tool never ran.

    A 4xx is Composio rejecting the call before execution (a 408 is a
    timeout and can arrive after it ran). A 5xx or an unreachable Composio
    leaves the result unknown.
    """
    status = exc.status
    return status is not None and 400 <= status < 500 and status != 408


def _created_file_result(
    service: Any,
    file_id: str,
    tool_data: Any,
    *,
    md5: str,
    size: int,
    action: str,
    target_mime_type: str | None = None,
) -> str | None:
    """The result for a create whose new id Drive returned.

    The file is read back and its checksum checked against the bytes that
    were staged, so a vendor that stored something else is not reported as a
    success. ``target_mime_type`` is the type the create was asked for; a
    Google Docs/Sheets/Slides conversion happens inside the create, so for
    such a target a file that came back as another type was not converted,
    and that is a failure rather than a success on matching bytes. A
    conversion target whose stored type Drive's own read-back did not
    establish is an unknown outcome, not a success: an id proves the file
    exists, not that it was converted, and a type the tool's answer claims is
    not what Drive stored. Any other target is not compared (see
    ``_needs_exact_create_type``): the type the tool sends there may be the
    enum's fallback or the staged object's real type, so a different label is
    not evidence that the create failed. The read-back fields beyond the
    native four are not returned.

    ``None`` means the read-back found no file with that id: nothing about
    this call is known, and the caller reports the same-name files it finds
    as candidates instead of reporting an id that does not resolve.
    """
    metadata, read_back = _created_file_metadata(service, file_id, tool_data)
    if metadata is None:
        return None
    mismatch = _created_content_mismatch(metadata, md5=md5, size=size)
    if mismatch is not None:
        return _content_update_error(
            f"Google Drive created a file with id '{file_id}', but the content "
            f"it stored is not the file that was {action} ({mismatch}). Do not "
            "upload it again before checking or removing that file: a retry "
            "would create a duplicate."
        )
    stored_mime_type = _normalize_mime_type(str(metadata.get("mimeType") or ""))
    if _needs_exact_create_type(target_mime_type):
        if not read_back or not stored_mime_type:
            # An id, or the tool's own claim about the type, proves neither
            # that nor which conversion happened.
            return _content_update_error(
                f"Google Drive has a file with id '{file_id}', but its type was "
                "not established from Google Drive's own read-back, so it is not "
                f"known whether the conversion to '{target_mime_type}' happened. "
                "The file was created; whether it holds the requested format is "
                "unverified. Check that file in Google Drive before creating it "
                "again: a retry would create a duplicate."
            )
        if stored_mime_type != target_mime_type:
            return _content_update_error(
                f"Google Drive created a file with id '{file_id}', but its type is "
                f"'{stored_mime_type}', not the requested '{target_mime_type}': the "
                "conversion to the requested type did not happen, so the file does "
                f"not hold the content that was {action} in that format. Do not "
                "create it again before checking or removing that file: a retry "
                "would create a duplicate."
            )
    return json.dumps(
        {"status": "success", "file": _created_file_output(metadata)},
        ensure_ascii=False,
    )


def _unconfirmed_create_result(
    service: Any,
    *,
    action: str,
    name: str,
    parent_id: str | None,
    reason: str | None = None,
) -> str:
    """Report what a create that could not be confirmed did, and what to check.

    The same-name lookup is diagnostic only: a file it finds is named as a
    candidate with its id, name and creation time, and the answer says the
    user has to check it. Nothing found by name, folder, time, type or
    checksum is reported as this call's success, because none of those ties
    the file to this execution.
    """
    candidates, lookup_error = _same_name_files(service, name=name, parent_id=parent_id)
    message = _unconfirmed_create_message(
        action=action,
        name=name,
        candidates=candidates,
        lookup_error=lookup_error,
        reason=reason,
    )
    diagnostic = [_create_candidate(candidate) for candidate in candidates]
    if diagnostic:
        return _content_update_error(message, candidates=diagnostic)
    return _content_update_error(message)


def _resource_key_refusal(action: str) -> str:
    """The refusal for a parent folder that needs a resource key.

    The Drive tool takes a bare parent id and cannot carry the key.
    ``action`` is "uploaded" or "created", so the answer says what did not
    happen.
    """
    return _content_update_error(
        "This Drive connection cannot create a file inside a folder that "
        "requires a resource key (a link-shared folder created before 2021): "
        "the Composio Drive tool cannot carry the key. Move the file to a "
        "folder shared with the connected Google account, or use a native "
        f"Google Drive connection. Nothing was {action}."
    )


def _run_composio_create(
    *,
    service: Any,
    staged: dict[str, str],
    name: str,
    tool_mime_type: str,
    parent_id: str | None,
    md5: str,
    size: int,
    action: str,
    target_mime_type: str | None = None,
) -> str:
    """Run the create tool on a staged object and report what Drive holds.

    ``action`` is "uploaded" or "created" and only colours the answer.
    ``target_mime_type`` is the type the tool was asked for; the read-back is
    compared with it only for a Docs/Sheets/Slides conversion target
    (``_needs_exact_create_type``).

    One rule settles a failed call for both callers: a definite Composio
    refusal (a 4xx other than 408), and an exception raised here before any
    request went out, sent nothing to Drive; a timeout, a 5xx or an
    unreachable Composio may arrive after the tool ran, so the outcome stays
    unknown and the same-name files are reported as candidates to check.
    """

    def _unconfirmed(reason: str | None = None) -> str:
        # A timed-out or unanswered tool may still be running in Composio, so
        # nothing here is reported as "not created". A same-name file only
        # becomes a candidate to check, never this call's result.
        return _unconfirmed_create_result(
            service, action=action, name=name, parent_id=parent_id, reason=reason
        )

    arguments: dict[str, Any] = {
        "name": name,
        "mimeType": tool_mime_type,
        "file_to_upload": staged,
    }
    if parent_id is not None:
        arguments["parents"] = [parent_id]
    what = "uploading" if action == "uploaded" else "creating"
    try:
        result = composio_tools.execute_tool(
            GOOGLEDRIVE_CREATE_FILE,
            arguments,
            toolkit=GOOGLEDRIVE_TOOLKIT_NAME,
        )
    except Exception as exc:
        logger.error(f"Error {what} file: {exc}")
        if not isinstance(exc, composio_tools.ComposioToolError) or (
            _composio_rejection_is_definite(exc)
        ):
            # Each verb keeps its own preposition: uploaded to, created in.
            return _content_update_error(
                f"The file was not {action} "
                f"{'to' if action == 'uploaded' else 'in'} Google Drive: "
                f"nothing was sent to Google Drive. {exc}"
            )
        # The request went out and its answer left the result open (a
        # timeout, a 5xx, an unreachable Composio, or a redirect that is not
        # followed). The tool's own words are passed on: they are not proof,
        # but they tell the user what to check, and they say the call was
        # sent rather than that nothing happened.
        return _unconfirmed(str(exc))
    if not result.successful:
        # Composio can fail the tool after Drive acted, so a failed answer is
        # no proof the file does not exist. Its own explanation is not proof
        # either, but it is what the user can act on.
        logger.error(f"Error {what} file: {result.error}")
        return _unconfirmed(result.error)
    file_id = _tool_file_object(result.data).get("id")
    if not isinstance(file_id, str) or not file_id:
        return _unconfirmed()
    outcome = _created_file_result(
        service,
        file_id,
        result.data,
        md5=md5,
        size=size,
        action=action,
        target_mime_type=target_mime_type,
    )
    return _unconfirmed() if outcome is None else outcome


def upload_file(
    *,
    service: Any,
    file_path: str,
    local_path: Path,
    name: str,
    mime_type: str,
    parent_id: str | None,
    parent_resource_key: str | None,
) -> str:
    """Upload a local file with Composio's create tool and the Files API.

    The proxy cannot carry a create's metadata and content in one request,
    so the bytes are staged and the tool names the staged object. Staging and
    execution are separate failures: a staging failure leaves Drive
    untouched, while a lost tool answer can mean the file exists, so the
    answer stays unknown and the same-name files already in that folder are
    listed as candidates to check (see ``_run_composio_create``).
    """
    if parent_resource_key is not None:
        return _resource_key_refusal("uploaded")
    try:
        fh_ctx = local_path.open("rb")
    except OSError as exc:
        logger.warning("Failed to open upload file %s: %s", local_path, exc)
        return _content_update_error("Could not read the file at the given path")
    with fh_ctx as fh:
        size = os.fstat(fh.fileno()).st_size
        if size == 0:
            return _content_update_error(f"File is empty: {file_path}")
        mtime_ns = _file_mtime_ns(fh)
        try:
            local_md5, local_bytes, block_digests = _hash_replacement_file(
                fh, size, keep_bytes=size <= _STAGED_UPLOAD_IN_MEMORY_BYTES
            )
        except ValueError as exc:
            return _content_update_error(_then_nothing_changed(exc))
        except OSError as exc:
            logger.warning("Failed to read upload file %s: %s", local_path, exc)
            return _content_update_error("Could not read the file at the given path")
        body: bytes | _VerifiedStagedReader = (
            local_bytes
            if local_bytes is not None
            else _VerifiedStagedReader(fh, size=size, block_digests=block_digests)
        )

        def _verify_unchanged_on_dedup() -> None:
            # Deduplication does not read the body, so nothing else would
            # notice a local change between the hash pass and the tool call.
            if _file_changed_since(fh, size=size, mtime_ns=mtime_ns):
                raise _ReplacementChangedError(_LOCAL_FILE_CHANGED)

        try:
            staged = composio_tools.stage_file(
                toolkit_slug=GOOGLEDRIVE_TOOLKIT_SLUG,
                tool_slug=GOOGLEDRIVE_CREATE_FILE,
                filename=name,
                mimetype=mime_type,
                md5=local_md5,
                data=body,
                length=size,
                on_dedup=(_verify_unchanged_on_dedup if local_bytes is None else None),
            )
        except _ReplacementChangedError:
            return _content_update_error(
                "The local file changed while it was being uploaded, so the "
                "upload was stopped before it completed and Google Drive was "
                f"not changed. {_LOCAL_FILE_CHANGED}"
            )
        except Exception as exc:
            logger.error(f"Error uploading file: {exc}")
            return _content_update_error(
                "The file was not uploaded to Google Drive: its bytes could "
                f"not be staged in Composio's file storage. {exc}"
            )
    return _run_composio_create(
        service=service,
        staged=staged,
        name=name,
        tool_mime_type=_create_tool_mime_type(mime_type),
        parent_id=parent_id,
        md5=local_md5,
        size=size,
        action="uploaded",
    )


def create_file(
    *,
    service: Any,
    name: str,
    content: str,
    mime_type: str,
    upload_mime_type: str,
    parent_id: str | None,
    parent_resource_key: str | None,
) -> str:
    """Create a text-backed file with Composio's create tool.

    Google's conversion from text to a Google Doc happens inside the create's
    multipart request, so the staged upload's type is the source type
    (``upload_mime_type``) while the tool's ``mimeType`` is the target. The
    failure split is the upload path's (see ``_run_composio_create``).
    """
    if parent_resource_key is not None:
        return _resource_key_refusal("created")
    data = content.encode("utf-8")
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    try:
        staged = composio_tools.stage_file(
            toolkit_slug=GOOGLEDRIVE_TOOLKIT_SLUG,
            tool_slug=GOOGLEDRIVE_CREATE_FILE,
            filename=name,
            mimetype=upload_mime_type,
            md5=md5,
            data=data,
            length=len(data),
        )
    except Exception as exc:
        logger.error(f"Error creating file: {exc}")
        return _content_update_error(
            "The file was not created in Google Drive: its content could not "
            f"be staged in Composio's file storage. {exc}"
        )
    # What the tool is asked to create. Only the tool's own enum can be sent,
    # so a caller's type outside it becomes application/octet-stream. Only a
    # Docs/Sheets/Slides target is then compared with the read-back: for any
    # other type which label Drive stores is not settled, so the bytes are
    # the evidence there (see _needs_exact_create_type).
    target_mime_type = _create_tool_mime_type(mime_type)
    return _run_composio_create(
        service=service,
        staged=staged,
        name=name,
        tool_mime_type=target_mime_type,
        parent_id=parent_id,
        md5=md5,
        size=len(data),
        action="created",
        target_mime_type=target_mime_type,
    )


def _unreadable_content_update_message(
    name: str, read_error: Exception | None, *, lead: str
) -> str:
    """Guidance for an update whose file cannot be read to settle its result.

    ``lead`` says what happened; this adds why the outcome is unknown and
    what the user should check. A read refused for a missing or inaccessible
    file gets the matching guidance instead of a plain transport message.
    """
    unknown = (
        f"so it is not known whether '{name}' changed. Do not report it as "
        "updated or as unchanged; ask the user to check the file's version "
        "history in Google Drive before trying again."
    )
    if read_error is not None and is_google_file_unavailable_error(read_error):
        return _unavailable_content_update_message(
            read_error,
            lead=(
                f"{lead}: the file no longer exists, or this Drive connection "
                f"can no longer access it, {unknown}"
            ),
        )
    if read_error is not None and is_google_file_access_error(read_error):
        return _content_permission_message(
            name,
            read_error,
            lead=(
                f"{lead}: the connected Google account was refused permission "
                f"to read '{name}' again to check it, {unknown}"
            ),
        )
    detail = f" {_content_error_detail(read_error)}" if read_error is not None else ""
    return f"{lead}, {unknown}{detail}"


def _composio_update_failure(
    *,
    service: Any,
    file_id: str,
    resource_key: str | None,
    name: str,
    current: dict[str, Any],
    current_summary: dict[str, Any],
    local_md5: str,
    local_size: int,
    stored_mime_type: str,
    reason: str,
    refusal: _Refusal = "reported_failure",
) -> str:
    """The result for an upload Composio reported as failed.

    ``refusal`` is what the failed call itself established. "definite" is
    Composio answering a 4xx other than 408: the tool never ran, so nothing
    was applied, and the answer is settled without a read-back. "unclear" is
    a timeout, a 5xx or an unreachable Composio: the write may still land
    after this returns, so an unchanged file only means "not yet".
    "reported_failure" is a tool that ran and answered that it failed: the
    read-back decides, and a file that still has the content read before this
    call is evidence that nothing changed.

    The file is read again through the proxy whichever the case; its checksum
    being the uploaded one means the upload landed, whatever the tool said.
    A file that changed without holding the new content leaves the outcome
    unknown as well.
    """
    now, read_error = _read_content_update_file(service, file_id, resource_key)
    if now is None:
        if refusal == "definite":
            # A definite refusal applied nothing, so its result needs no
            # read-back. The pre-read state stays "previous": the file was
            # not read again to show that it is still in that state.
            return _content_update_error(
                f"Google Drive did not accept the new content for '{name}'. "
                f"Do not report the file as updated. {_NOTHING_CHANGED} {reason}",
                previous=current_summary,
            )
        return _content_update_error(
            _unreadable_content_update_message(
                name,
                read_error,
                lead=(
                    "The upload to Google Drive did not finish normally, and "
                    "the file could not be read again to check it"
                ),
            ),
            previous=current_summary,
        )
    if _holds_content(now, md5=local_md5, size=local_size):
        return _content_update_success(
            updated=now,
            current=current,
            current_summary=current_summary,
            name=name,
            stored_mime_type=stored_mime_type,
            note=(
                " Composio reported an error for the upload, but Google "
                "Drive's own checksum shows exactly the new content."
            ),
        )
    if not _same_content(now, current):
        return _content_update_error(
            f"Google Drive did not accept the new content for '{name}', and "
            "the file has changed since it was read before this call, so it "
            "is not known whether this call changed it. Do not report it as "
            f"updated; ask the user to check the file's version history. {reason}",
            previous=current_summary,
        )
    if refusal == "unclear":
        return _content_update_error(
            "The upload to Google Drive did not finish normally, and the file "
            "still has the same content and head revision it had before this "
            f"call, so it is not known whether '{name}' changed: the upload "
            "may still be finishing in Composio. Do not report it as updated "
            "or as unchanged; ask the user to check the file's version "
            f"history in Google Drive before trying again. {reason}",
            previous=current_summary,
        )
    return _content_update_error(
        f"Google Drive did not accept the new content for '{name}'. Do not "
        f"report the file as updated. {_NOTHING_CHANGED} {reason}",
        file=_content_summary(now),
    )


def replace_content(
    *,
    service: Any,
    file_id: str,
    resource_key: str | None,
    name: str,
    current: dict[str, Any],
    current_summary: dict[str, Any],
    fh: BinaryIO,
    local_size: int,
    local_md5: str,
    local_bytes: bytes | None,
    block_digests: list[bytes],
    mtime_ns: int,
    upload_mime_type: str,
    stored_mime_type: str,
) -> str:
    """Replace a file's content with Composio's media-only update tool.

    The pre-read, the identical-content short circuit and the head revision
    check have already run in the caller; this stages the bytes, runs the
    tool, and then reads the file back through the proxy to decide the
    result. A single media upload is what Composio's tool carries, so this
    path is not limited by Proxy Execute's JSON body size.
    """
    if resource_key is not None:
        # The tool takes a bare fileId and cannot carry the key, so refusing
        # here is the same refusal the upload and create paths make -- and it
        # comes before the bytes are staged.
        return _content_update_error(
            "This Drive connection cannot replace the content of a file that "
            "requires a resource key (a link-shared file created before "
            "2021): the Composio Drive tool cannot carry the key. Copy the "
            "file into a folder shared with the connected Google account and "
            "update that copy, or use a native Google Drive connection. "
            "Nothing was changed.",
            file=current_summary,
        )
    body: bytes | _VerifiedStagedReader = (
        local_bytes
        if local_bytes is not None
        else _VerifiedStagedReader(fh, size=local_size, block_digests=block_digests)
    )

    def _verify_unchanged_on_dedup() -> None:
        # Deduplication does not read the body, so nothing else would notice
        # a local change between the hash pass and the tool call.
        if _file_changed_since(fh, size=local_size, mtime_ns=mtime_ns):
            raise _ReplacementChangedError(_LOCAL_FILE_CHANGED)

    try:
        staged = composio_tools.stage_file(
            toolkit_slug=GOOGLEDRIVE_TOOLKIT_SLUG,
            tool_slug=GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
            filename=name,
            mimetype=upload_mime_type,
            md5=local_md5,
            data=body,
            length=local_size,
            on_dedup=(_verify_unchanged_on_dedup if local_bytes is None else None),
        )
    except _ReplacementChangedError:
        return _content_update_error(
            "The local file changed while it was being uploaded, so the "
            "upload was stopped before it completed and Google Drive kept "
            "the file as it was. Finish writing the replacement file before "
            f"calling this tool. {_NOTHING_CHANGED}",
            file=current_summary,
        )
    except Exception as exc:
        logger.error(f"Error staging replacement content: {exc}")
        return _content_update_error(
            "The replacement content could not be uploaded to Composio's "
            f"file storage, so Google Drive was not changed. {exc}",
            file=current_summary,
        )
    try:
        result = composio_tools.execute_tool(
            GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
            {
                "fileId": file_id,
                "file_to_upload": staged,
                "uploadType": "media",
                "supportsAllDrives": True,
            },
            toolkit=GOOGLEDRIVE_TOOLKIT_NAME,
        )
    except Exception as exc:
        logger.error(f"Error updating file content: {exc}")
        # A 4xx other than 408 is Composio refusing the call before the tool
        # ran, so an unchanged file is a definite "nothing changed". A
        # timeout, a 5xx, an unreachable Composio or anything else may arrive
        # after the write landed or before it does, so an unchanged file is
        # only "not yet" and the outcome stays unknown.
        definite = isinstance(exc, composio_tools.ComposioToolError) and (
            _composio_rejection_is_definite(exc)
        )
        return _composio_update_failure(
            service=service,
            file_id=file_id,
            resource_key=resource_key,
            name=name,
            current=current,
            current_summary=current_summary,
            local_md5=local_md5,
            local_size=local_size,
            stored_mime_type=stored_mime_type,
            reason=f"Composio reported: {exc}",
            refusal="definite" if definite else "unclear",
        )
    if not result.successful:
        reason = (
            f"Composio reported: {result.error}"
            if result.error
            else "Composio reported that the upload failed without a reason."
        )
        return _composio_update_failure(
            service=service,
            file_id=file_id,
            resource_key=resource_key,
            name=name,
            current=current,
            current_summary=current_summary,
            local_md5=local_md5,
            local_size=local_size,
            stored_mime_type=stored_mime_type,
            reason=reason,
        )
    updated, read_error = _read_content_update_file(service, file_id, resource_key)
    if updated is None:
        return _content_update_error(
            _unreadable_content_update_message(
                name,
                read_error,
                lead=(
                    "Google Drive accepted the upload, but the file could not "
                    "be read again to confirm the new content"
                ),
            ),
            previous=current_summary,
        )
    mismatch = _content_update_mismatch(
        updated,
        current,
        file_id=file_id,
        local_md5=local_md5,
    )
    if mismatch is not None:
        return _content_update_error(
            f"Google Drive accepted the upload for '{name}', but the file it "
            f"returned does not confirm the new content ({mismatch}). Do "
            "not report this as done; ask the user to check the file's "
            "version history in Google Drive.",
            file=_content_summary(updated),
            previous=current_summary,
        )
    return _content_update_success(
        updated=updated,
        current=current,
        current_summary=current_summary,
        name=name,
        stored_mime_type=stored_mime_type,
    )
