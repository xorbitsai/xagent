from __future__ import annotations

import mimetypes
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import quote, unquote, urlsplit

WORKSPACE_OUTPUT_FILES_TOOL_NAME = "get_workspace_output_files"

# Minted inside the sandbox runner, which reaches no database. Only the runner
# that minted one resolves it, through its own path cache; the host re-registers
# these after the call, so one surviving on the host means that failed.
SANDBOX_FILE_ID_PREFIX = "sandbox-"


def is_sandbox_local_file_id(file_id: str | None) -> bool:
    return bool(file_id) and str(file_id).startswith(SANDBOX_FILE_ID_PREFIX)


_FINAL_DELIVERABLE_FILE_LOOKUP_INSTRUCTIONS = f"""If a generated final deliverable exists but no trusted file_id remains in the current
context, call {WORKSPACE_OUTPUT_FILES_TOOL_NAME} once before finalizing. Match the exact
deliverable path and filename, then use only a returned non-null file_id and render [filename](file:file_id)
with the exact returned filename; this matches the markdown_link form.
If the lookup has no registered file_id, do not repeat the lookup, invent a link,
or claim delivery.

"""

_FINAL_DELIVERABLE_NO_LOOKUP_INSTRUCTIONS = """If no trusted link or prescribed rendering remains in context and lookup is unavailable,
do not reconstruct one or claim delivery; state that the deliverable link is unavailable.

"""


def final_deliverable_file_reference_instructions(
    *, can_lookup: bool, include_heading: bool = True
) -> str:
    """Build final-deliverable rules for the current prompt location.

    Final-only protocol turns cannot call workspace tools. Omitting lookup advice
    there prevents the prompt from requiring a tool call that the active schema
    rejects. Schema descriptions also omit the Markdown heading because they
    embed these rules inline after existing prose.
    """
    lookup_instructions = (
        _FINAL_DELIVERABLE_FILE_LOOKUP_INSTRUCTIONS
        if can_lookup
        else _FINAL_DELIVERABLE_NO_LOOKUP_INSTRUCTIONS
    )
    heading = "## FINAL DELIVERABLE FILE REFERENCES\n" if include_heading else ""
    return f"""{heading}If a successful tool result produced a file (or a trusted non-internal FileRef references one)
that satisfies the user's requested final deliverable, the final answer MUST include
the exact markdown_link returned for that file, unless the successful tool result
prescribes a different exact user-facing rendering, such as inline_markdown for screenshots,
or the FileRef rules require inline image Markdown for an image intended for inline display.

{lookup_instructions}Copy the selected rendering verbatim and preserve its original filename and extension.
Preserve every returned file_id exactly.

Include only user-requested final deliverables. Do not include intermediate,
supporting, or temporary files unless the user explicitly requested them as
deliverables. Never include internal FileRefs in user-facing output."""


FILE_REF_OUTPUT_INSTRUCTIONS = """## FILE REFERENCE OUTPUTS
When mentioning a generated or uploaded file that has a file_id, render it as a Markdown file reference:
- Files: [filename](file:file_id)
- Images intended for inline display: ![filename](file:file_id)
Do not mention only the filename when a file_id or markdown_link is available. Prefer an existing markdown_link value when one is present.

File delivery integrity:
- Internal FileRefs (`internal: true`) are runtime context only. Never render or expose a `file:` link for them.
- Never invent, guess, or construct a file_id or `file:` link. Use only a trusted FileRef supplied for an existing input/attachment, or the exact FileRef or markdown_link returned by a successful tool result. A link mentioned only in prior assistant prose is not provenance.
- An exact non-null file_id and filename returned together by get_workspace_output_files are trusted provenance for rendering a file link.
- When the user requests a new file or file-based artifact, it is not delivered until a successful tool result returns its registered FileRef or markdown_link.
- Do not call final_answer claiming that a file was created or delivered unless that result exists."""

FILE_REF_OUTPUT_INSTRUCTIONS += """
- Tool execution success and file validation are separate. If validation.status is invalid, repair the file and recheck it, or clearly report the failure; do not present it as a usable completed deliverable. Keep its file_id available for repair.
- An unchecked or absent validation result is not a pass. State that limitation when delivering the file; a valid result establishes format readability only, not business/content correctness."""

FILE_REF_MODEL_INSTRUCTIONS = f"""## FILE REFERENCES
Files are referenced by FileRef objects. Treat file_id as the canonical file handle.

Rules:
- Use file_id when reading files or passing files to tools.
- Do not guess storage paths such as /uploads/... or user_id/... paths.
- For HTML assets, call prepare_html_asset(file_id, html_path, alias) first.
- Use the returned html_src inside HTML, CSS, script, or image references.
- For user-visible output links, follow the file reference output rules below.

{FILE_REF_OUTPUT_INSTRUCTIONS}"""


def parse_file_id_ref(value: str | None) -> str | None:
    """Extract a file id from an internal ``file:`` reference.

    ``file:<id>`` is the canonical Xagent form. ``file://<id>`` is accepted
    for compatibility with older chat and connector messages. Real file URIs
    such as ``file:///absolute/path`` are deliberately not treated as file
    ids; callers must still route paths through workspace containment checks.
    """
    if value is None:
        return None

    raw = str(value).strip()
    if not raw.startswith("file:"):
        return None

    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    if parsed.scheme != "file" or parsed.query or parsed.fragment:
        return None

    if raw.startswith("file://"):
        # ``file://<id>`` places the legacy id in the URI authority. A path
        # means this is a real/remote file URI instead of an internal handle.
        if not parsed.netloc or parsed.path:
            return None
        candidate = parsed.netloc
    else:
        candidate = parsed.path

    file_id = unquote(candidate).strip()
    if not file_id or file_id in {".", ".."}:
        return None
    if "/" in file_id or "\\" in file_id:
        return None
    return file_id


def build_file_id_ref(file_id: str) -> str:
    """Return the canonical internal URI for a registered file id."""
    normalized = str(file_id).strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError("file_id must be a non-empty identifier")
    decoded = unquote(normalized)
    if decoded in {".", ".."} or any(
        separator in value for value in (normalized, decoded) for separator in "/\\"
    ):
        raise ValueError("file_id must not contain path separators")
    return f"file:{quote(normalized, safe='')}"


def guess_mime_type(filename: str) -> str:
    media_type, _ = mimetypes.guess_type(filename)
    return media_type or "application/octet-stream"


def build_file_ref(
    *,
    file_id: str | None,
    filename: str,
    mime_type: str | None = None,
    size: int | None = None,
    internal: bool = False,
) -> dict[str, Any]:
    """Build the model/API-facing file reference for a registered file."""
    resolved_mime_type = mime_type or guess_mime_type(filename)
    result: dict[str, Any] = {
        "file_id": file_id,
        "filename": filename,
        "mime_type": resolved_mime_type,
    }
    if size is not None:
        result["size"] = int(size)
    if internal:
        result["internal"] = True

    if file_id and not internal:
        encoded_file_id = quote(file_id, safe="")
        result.update(
            {
                "preview_url": f"/api/files/preview/{encoded_file_id}",
                "download_url": f"/api/files/download/{encoded_file_id}",
                "markdown_link": f"[{filename}]({build_file_id_ref(file_id)})",
            }
        )
    else:
        result.update(
            {
                "preview_url": None,
                "download_url": None,
                "markdown_link": None,
            }
        )
    return result


def build_workspace_file_ref(
    *,
    workspace: Any,
    file_path: str | Path,
    file_id: str | None = None,
    mime_type: str | None = None,
    internal: bool = False,
    validate: bool = False,
) -> dict[str, Any]:
    """Register a workspace file and build the model/API-facing FileRef.

    ``internal`` keeps execution scratch data resolvable by ``file_id`` without
    creating a user-visible or durably uploaded file record. It fails closed
    when the workspace does not support internal registration.

    ``validate`` checks a completed output snapshot without changing registration
    or tool success. Call from a worker thread, as with other blocking file I/O.
    It applies only to user-visible outputs; internal scratch files are not checked.
    """
    resolved_path = Path(file_path).resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"File not found for FileRef: {file_path}")
    if not hasattr(workspace, "workspace_dir"):
        raise ValueError("Workspace does not expose workspace_dir")

    if internal:
        register_internal = getattr(workspace, "register_internal_file", None)
        if not callable(register_internal):
            raise TypeError("workspace does not support internal file registration")
        final_file_id = register_internal(str(resolved_path))
    else:
        final_file_id = file_id or workspace.get_file_id_from_path(str(resolved_path))
        if not final_file_id:
            final_file_id = workspace.register_file(str(resolved_path))

    workspace_root = workspace.workspace_dir.resolve()
    file_ref = build_file_ref(
        file_id=final_file_id,
        filename=resolved_path.name,
        mime_type=mime_type,
        size=resolved_path.stat().st_size,
        internal=internal,
    )
    result = {
        **file_ref,
        "file_path": str(resolved_path),
    }
    if not internal:
        try:
            relative_path = str(resolved_path.relative_to(workspace_root))
        except ValueError:
            relative_path = str(resolved_path)
        result["relative_path"] = relative_path
        if validate:
            from .artifact_validation.service import validate_artifact

            result["validation"] = validate_artifact(resolved_path).as_dict()
    return result


def sanitize_file_ref_for_context(file_ref: dict[str, Any]) -> dict[str, Any]:
    """Return the durable, model-safe subset of a registered FileRef.

    Local storage paths are intentionally excluded so messages, checkpoints,
    and traces can persist this value without leaking host filesystem details.
    """

    file_id = str(file_ref.get("file_id") or "").strip()
    raw_filename = str(file_ref.get("filename") or "").strip()
    filename = Path(raw_filename.replace("\\", "/")).name.strip()
    if not file_id:
        raise ValueError("FileRef must contain a registered file_id")
    if not filename:
        raise ValueError("FileRef must contain a filename")
    build_file_id_ref(file_id)

    mime_type = str(file_ref.get("mime_type") or guess_mime_type(filename))
    raw_size = file_ref.get("size")
    size = int(raw_size) if raw_size is not None else None
    if size is not None and size < 0:
        raise ValueError("FileRef size must not be negative")
    internal = file_ref.get("internal") is True
    result = build_file_ref(
        file_id=file_id,
        filename=filename,
        mime_type=mime_type,
        size=size,
        internal=internal,
    )
    relative_path = file_ref.get("relative_path")
    if not internal and relative_path is not None:
        raw_relative = str(relative_path).strip()
        windows_relative = PureWindowsPath(raw_relative)
        relative = Path(raw_relative.replace("\\", "/"))
        is_windows_rooted = bool(windows_relative.drive or windows_relative.root)
        if (
            raw_relative
            and not is_windows_rooted
            and not relative.is_absolute()
            and ".." not in relative.parts
        ):
            result["relative_path"] = relative.as_posix()
    return result


def safe_asset_filename(filename: str) -> str:
    """Return a browser-safe basename for HTML bundle assets."""
    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        return "asset"
    return name
