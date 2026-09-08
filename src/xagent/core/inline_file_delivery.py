"""Turn explicit encoded-file deliveries into registered workspace attachments.

This is a final-answer delivery boundary, not a general Base64 detector or a
format validator. Intermediate messages and execution checkpoints are unchanged.
Only data-URI Markdown links outside code and explicitly named Base64 fences
opt in. Arbitrary prose, source code and unnamed encoded examples are preserved.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import get_inline_file_delivery_max_bytes
from .file_ref import build_workspace_file_ref

logger = logging.getLogger(__name__)
_MAX_FILES_PER_RUN = 8
_DATA_PARAMETERS = r"(?:;[\w!#$%&'*+.^`|~-]+=[^;\s,()\[\]]*)*"

# Never infer active HTML/SVG or executable types from untrusted model output.
_EXTENSIONS = {
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/tab-separated-values": ".tsv",
    "application/json": ".json",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_DATA_LINK = re.compile(
    r"(?P<image>!)?\[(?P<label>[^\[\]\r\n]*)\]\("
    r"data:(?P<mime>[\w.+/-]+)" + _DATA_PARAMETERS + r";base64,"
    r"(?P<payload>[^\s()\[\]]*)(?P<closed>\))?",
    re.IGNORECASE,
)
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)")
_STREAM_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})([^\r\n]*)")
_NAMED_BASE64 = re.compile(r"base64[ \t]+filename=([^\r\n]+)\Z", re.IGNORECASE)
_STREAM_MARKER = re.compile(
    r"\]\(data:[\w.+/-]+" + _DATA_PARAMETERS + r";base64,", re.IGNORECASE
)
_LIST_ITEM = re.compile(r"^( *)(?:[-+*]|\d{1,9}[.)])[ \t]+")
_DATA_CONTINUATION = re.compile(
    r"[ \t]*(?P<payload>[A-Za-z0-9+/=]*)[ \t]*(?P<closed>\))?"
)


class InlineFileStreamGuard:
    """Hold a possible encoded tail until the authoritative final replacement."""

    def __init__(self) -> None:
        self.pending = ""
        self.held = False
        self._at_line_start = True

    def feed(self, delta: str) -> str:
        if self.held:
            return ""
        self.pending += delta
        marker = _STREAM_MARKER.search(self.pending)
        marker_start = marker.start() if marker else None
        offset = 0
        for line in self.pending.splitlines(keepends=True):
            fence = _STREAM_FENCE.match(line) if offset or self._at_line_start else None
            if fence and _NAMED_BASE64.fullmatch(fence.group(2).strip()):
                marker_start = (
                    min(marker_start, offset) if marker_start is not None else offset
                )
                break
            offset += len(line)
        if marker_start is not None:
            prefix = self.pending[:marker_start]
            self.pending = ""
            self.held = True
            return prefix
        # Retain enough suffix to recognize markers split across any chunk edge.
        keep = 6
        # A fence header can contain whitespace or a longer delimiter. Retain
        # its incomplete line until its language is known, not ordinary prose.
        fence = re.search(
            r"(?:^|\n)[ \t]*(?:`{1,2}|~{1,2}|`{3,}[^\r\n]*|~{3,}[^\r\n]*)$",
            self.pending,
        )
        if fence and (
            fence.start() or self._at_line_start or self.pending.startswith("\n")
        ):
            keep = max(keep, len(self.pending) - fence.start())
        # Retain a split data-URI header only until its comma disambiguates
        # Base64 from ordinary data URLs. Non-Base64 URLs keep streaming.
        header = re.search(r"\]\(data:[^,\s()\[\]]*$", self.pending, re.IGNORECASE)
        if header:
            keep = max(keep, len(self.pending) - header.start())
        prefix, self.pending = self.pending[:-keep], self.pending[-keep:]
        if prefix:
            self._at_line_start = prefix.endswith("\n")
        return prefix

    def flush(self) -> str:
        pending, self.pending = self.pending, ""
        return pending


class InlineFileDelivery:
    """Per-run bounded, idempotent delivery; all I/O runs off the event loop."""

    def __init__(self, workspace: Any) -> None:
        self.workspace = workspace
        self._lock = Lock()
        self._refs: dict[tuple[str, str], dict[str, Any]] = {}
        self._bytes = 0

    def transform(self, content: str) -> str:
        if "base64" not in content.lower():
            return content
        with self._lock:
            return self._transform(content)

    def _transform(self, content: str) -> str:
        lines = content.splitlines(keepends=True)
        output: list[str] = []
        i = 0
        inline_ticks = 0
        list_indents: list[int] = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                inline_ticks = 0
            indent = len(line) - len(line.lstrip(" "))
            if line.strip() and not inline_ticks:
                while list_indents and indent < list_indents[-1]:
                    list_indents.pop()
                item = _LIST_ITEM.match(line)
                parent_indent = list_indents[-1] if list_indents else 0
                if item and indent < parent_indent + 4:
                    list_indents.append(item.end())
            content_indent = list_indents[-1] if list_indents else 0
            fence_indent = content_indent if indent >= content_indent else 0
            fence = _FENCE.match(line[fence_indent:]) if not inline_ticks else None
            if fence:
                marker, info = fence.groups()
                end = i + 1
                closing = re.compile(
                    r"^ {0,3}"
                    + re.escape(marker[0])
                    + "{"
                    + str(len(marker))
                    + r",}[ \t]*(?:\r?\n)?$"
                )
                while end < len(lines) and not closing.fullmatch(
                    lines[end][fence_indent:]
                ):
                    end += 1
                named = _NAMED_BASE64.fullmatch(info.strip())
                if named:
                    filename = named.group(1).strip().strip('"')
                    suffix = Path(filename).suffix.lower()
                    if suffix == ".jpeg":
                        suffix = ".jpg"
                    mime = next(
                        (m for m, ext in _EXTENSIONS.items() if suffix == ext),
                        "",
                    )
                    replacement = self._deliver(
                        filename,
                        mime,
                        "".join(lines[i + 1 : end]) if end < len(lines) else "",
                    )
                    output.append(
                        line[:fence_indent]
                        + replacement
                        + (
                            "\n"
                            if end < len(lines) and lines[end].endswith("\n")
                            else ""
                        )
                    )
                else:
                    output.extend(lines[i : min(end + 1, len(lines))])
                i = end + 1
                continue
            if (
                indent >= content_indent + 4
                or line.startswith("\t")
                or re.match(r"^\s*>", line)
            ):
                # Decline indented code/quoted examples, including nested fences.
                output.append(line)
                i += 1
                continue
            # Scan code delimiters and links together, so a backtick inside an
            # explicit link label cannot accidentally turn its payload into code.
            pieces: list[str] = []
            pos = 0
            while pos < len(line):
                if line[pos] == "`":
                    end = pos + 1
                    while end < len(line) and line[end] == "`":
                        end += 1
                    ticks = end - pos
                    if not inline_ticks:
                        # Only matched delimiters open inline code. Code spans
                        # may cross soft line breaks, but not blank-line/fence
                        # boundaries; an unmatched tick is ordinary text.
                        paragraph = line[end:]
                        following = i + 1
                        while following < len(lines):
                            candidate = lines[following]
                            if not candidate.strip() or _FENCE.match(candidate):
                                break
                            paragraph += candidate
                            following += 1
                        if re.search(r"(?<!`)`{" + str(ticks) + r"}(?!`)", paragraph):
                            inline_ticks = ticks
                    elif inline_ticks == ticks:
                        inline_ticks = 0
                    pieces.append(line[pos:end])
                    pos = end
                    continue
                match = (
                    _DATA_LINK.match(line, pos)
                    if not inline_ticks and line[pos] in "!["
                    else None
                )
                if match and (pos == 0 or line[pos - 1] != "\\"):
                    payload = match["payload"]
                    closed = bool(match["closed"])
                    pos = match.end()
                    # A reflowed payload is still one explicit delivery. Consume
                    # only Base64 continuation lines, leaving following prose,
                    # code blocks and independent links to the normal scanner.
                    while not closed and not line[pos:].strip() and i + 1 < len(lines):
                        following_line = lines[i + 1]
                        continuation = _DATA_CONTINUATION.match(following_line)
                        assert continuation is not None
                        if not continuation["closed"] and (
                            not continuation["payload"]
                            or following_line[continuation.end() :].strip()
                        ):
                            break
                        payload += continuation["payload"]
                        closed = bool(continuation["closed"])
                        i += 1
                        line = following_line
                        pos = continuation.end()
                    pieces.append(
                        self._deliver(
                            match["label"],
                            match["mime"].lower(),
                            payload if closed else "",
                            image=bool(match["image"]),
                        )
                    )
                else:
                    pieces.append(line[pos])
                    pos += 1
            output.append("".join(pieces))
            i += 1
        return "".join(output)

    def _deliver(
        self, label: str, mime: str, encoded: str, *, image: bool = False
    ) -> str:
        extension = _EXTENSIONS.get(mime)
        # Labels are never paths or authority to select an executable suffix.
        name = label.replace("\\", "/").rsplit("/", 1)[-1]
        name = (
            "".join(c for c in name if c.isalnum() or c in " ._-()").strip(" .")[:100]
            or "attachment"
        )
        if extension and not name.lower().endswith(extension):
            name = (Path(name).stem or "attachment") + extension
        if extension:
            stem = name[: -len(extension)]
            name = (
                stem.encode("utf-8")[: 240 - len(extension)].decode(
                    "utf-8", errors="ignore"
                )
                + name[-len(extension) :]
            )
        unavailable = f"{name} (attachment unavailable)"
        if not extension:
            return unavailable
        try:
            limit = get_inline_file_delivery_max_bytes()
            max_encoded = ((limit + 2) // 3) * 4
            # Allow ordinary MIME wrapping (including CRLF) without removing
            # the bound on whitespace-heavy input before normalization.
            if limit <= 0 or len(encoded) > max_encoded + max_encoded // 16 + 4096:
                return unavailable
            encoded = re.sub(r"[\r\n\t ]", "", encoded)
            if len(encoded) > max_encoded:
                return unavailable
            data = base64.b64decode(encoded, validate=True)
            if not data or len(data) > limit:
                return unavailable
            key = (mime, hashlib.sha256(data).hexdigest())
            ref = self._refs.get(key)
            if ref is None:
                if (
                    len(self._refs) >= _MAX_FILES_PER_RUN
                    or self._bytes + len(data) > limit
                ):
                    return unavailable
                root = Path(self.workspace.workspace_dir).resolve()
                output = Path(self.workspace.output_dir).resolve()
                if not output.is_relative_to(root):
                    return unavailable
                output.mkdir(parents=True, exist_ok=True)
                # These are durable workspace outputs, not disposable temp
                # files. A unique directory avoids overwriting earlier runs.
                directory = Path(tempfile.mkdtemp(prefix="inline-", dir=output))
                path = directory / name
                try:
                    with path.open("xb") as stream:
                        stream.write(data)
                    register_delivery = getattr(
                        self.workspace, "register_delivery_file", None
                    )
                    if not callable(register_delivery):
                        raise ValueError("Workspace does not support durable delivery")
                    file_id = register_delivery(str(path))
                    if not file_id:
                        raise ValueError("Attachment registration returned no file ID")
                    ref = build_workspace_file_ref(
                        workspace=self.workspace,
                        file_path=path,
                        mime_type=mime,
                        file_id=file_id,
                    )
                    if not ref.get("markdown_link"):
                        raise ValueError("Attachment registration returned no link")
                except Exception:
                    logger.exception("Inline file registration failed")
                    path.unlink(missing_ok=True)
                    directory.rmdir()
                    return unavailable
                self._refs[key] = ref
                self._bytes += len(data)
            link = str(ref["markdown_link"])
            return "!" + link if image and mime.startswith("image/") else link
        except (ValueError, binascii.Error):
            return unavailable
        except Exception:
            logger.exception("Inline file delivery failed")
            return unavailable
