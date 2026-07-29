from __future__ import annotations

import html
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from ....core.file_ref import parse_file_id_ref


@dataclass(frozen=True)
class SlackFileRef:
    file_id: str
    label: str
    is_image: bool = False


_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]\n]*)\]\(([^)\n]+)\)")
_MARKDOWN_FILE_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\n]+)\)")


def strip_slack_file_refs(text: str) -> tuple[str, list[SlackFileRef]]:
    """Remove internal file links that should be uploaded through Slack."""
    if not text:
        return "", []

    refs: list[SlackFileRef] = []

    def replace_image(match: re.Match[str]) -> str:
        file_id = _extract_local_file_id(match.group(2))
        if not file_id:
            return match.group(0)
        refs.append(
            SlackFileRef(
                file_id=file_id,
                label=match.group(1).strip() or "image",
                is_image=True,
            )
        )
        return ""

    def replace_file(match: re.Match[str]) -> str:
        file_id = _extract_local_file_id(match.group(2))
        if not file_id:
            return match.group(0)
        refs.append(
            SlackFileRef(
                file_id=file_id,
                label=match.group(1).strip() or "file",
            )
        )
        return ""

    cleaned = _MARKDOWN_IMAGE_RE.sub(replace_image, text)
    cleaned = _MARKDOWN_FILE_LINK_RE.sub(replace_file, cleaned)

    deduped: list[SlackFileRef] = []
    seen_file_ids: set[str] = set()
    for ref in refs:
        if ref.file_id in seen_file_ids:
            continue
        seen_file_ids.add(ref.file_id)
        deduped.append(ref)

    return _clean_stripped_markdown_refs(cleaned), deduped


def markdown_to_slack(text: str) -> str:
    """Convert the small Markdown subset emitted by agents to Slack mrkdwn."""
    if not text:
        return ""

    escaped = html.escape(text, quote=False)
    escaped = re.sub(
        r"^[ \t]*#{1,6}\s+(.+)$",
        lambda match: f"*{match.group(1)}*",
        escaped,
        flags=re.MULTILINE,
    )
    escaped = re.sub(r"\*\*([^*\n]+)\*\*", r"*\1*", escaped)
    escaped = re.sub(r"__([^_\n]+)__", r"*\1*", escaped)
    escaped = re.sub(r"~~([^~\n]+)~~", r"~\1~", escaped)

    def replace_link(match: re.Match[str]) -> str:
        label = match.group(1)
        url = html.unescape(match.group(2))
        return f"<{url}|{label}>"

    return _MARKDOWN_LINK_RE.sub(replace_link, escaped)


def _extract_local_file_id(target: str) -> str | None:
    normalized_target = html.unescape(target.strip())
    internal_file_id = parse_file_id_ref(normalized_target)
    if internal_file_id is not None:
        return internal_file_id

    parsed = urlparse(normalized_target)
    for prefix in ("/api/files/preview/", "/api/files/download/"):
        if parsed.path.startswith(prefix):
            candidate = parsed.path.removeprefix(prefix).strip("/")
            return candidate or None
    return None


def _clean_stripped_markdown_refs(text: str) -> str:
    text = re.sub(r"(?m)^[ \t]*[-*•][ \t]*(?:\n|$)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
