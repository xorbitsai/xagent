import html
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from ....core.file_ref import parse_file_id_ref

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ....core.agent.service import AgentService


@dataclass(frozen=True)
class TelegramImageRef:
    file_id: str
    alt_text: str


@dataclass(frozen=True)
class TelegramFileRef:
    file_id: str
    label: str


def markdown_to_tg_html(text: str) -> str:
    """Convert basic Markdown to Telegram-supported HTML."""
    if not text:
        return ""

    # First, escape HTML special characters to prevent parsing errors
    text = html.escape(text)

    # Replace code blocks: ```lang\ncode\n```
    # We use <pre><code class="language-lang">...</code></pre> for Telegram
    def replace_code_block(match: re.Match) -> str:
        lang = match.group(1).strip()
        code = match.group(2)
        if lang:
            return f'<pre><code class="language-{lang}">{code}</code></pre>'
        return f"<pre>{code}</pre>"

    text = re.sub(r"```(.*?)\n(.*?)\n```", replace_code_block, text, flags=re.DOTALL)
    text = re.sub(r"```(.*?)```", r"<pre>\1</pre>", text, flags=re.DOTALL)

    text = _format_markdown_tables_for_telegram(text)

    # Replace inline code: `code`
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)

    # Replace headers: # Header or ## Header -> <b>Header</b>
    text = re.sub(r"^[ \t]*#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # Replace blockquotes: > quote (escaped to &gt; by html.escape)
    text = re.sub(
        r"^[ \t]*&gt;\s+(.+)$", r"<blockquote>\1</blockquote>", text, flags=re.MULTILINE
    )

    # Replace unordered lists: - item or * item -> • item
    # Also preserve the leading indentation
    text = re.sub(r"^([ \t]*)[\*\-]\s+(.+)$", r"\1• \2", text, flags=re.MULTILINE)

    # Replace bold: **bold** or __bold__
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__([^_\n]+)__", r"<b>\1</b>", text)

    # Replace italic: *italic* or _italic_
    # Be careful not to match inside words like snake_case, and don't cross newlines
    text = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^\_\n]+)_(?!\w)", r"<i>\1</i>", text)

    # Replace strikethrough: ~~strike~~
    text = re.sub(r"~~([^~\n]+)~~", r"<s>\1</s>", text)

    # Replace links: [text](url)
    text = re.sub(r"\[([^\]\n]+)\]\(([^)\n]+)\)", r'<a href="\2">\1</a>', text)

    return text


def _format_markdown_tables_for_telegram(text: str) -> str:
    """Render Markdown tables as wrapped Telegram-friendly lists."""
    lines = text.splitlines()
    rendered_lines: list[str] = []
    index = 0

    while index < len(lines):
        if _is_markdown_table_start(lines, index):
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            rendered_lines.extend(_render_markdown_table_as_list(table_lines))
            continue

        rendered_lines.append(lines[index])
        index += 1

    return "\n".join(rendered_lines)


def _is_markdown_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and "|" in lines[index]
        and _is_markdown_table_separator(lines[index + 1])
    )


def _is_markdown_table_separator(line: str) -> bool:
    cells = _split_markdown_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-+:?", cell) for cell in cells)


def _split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _render_markdown_table_as_list(table_lines: list[str]) -> list[str]:
    headers = _split_markdown_table_row(table_lines[0])
    rows = [
        _split_markdown_table_row(line)
        for line in table_lines[2:]
        if line.strip() and "|" in line
    ]
    if not headers or not rows:
        return table_lines

    rendered: list[str] = []
    for row in rows:
        padded_row = row + [""] * max(0, len(headers) - len(row))
        if len(headers) == 2:
            title = padded_row[0] or headers[0]
            detail = padded_row[1]
            rendered.append(
                f"• <b>{title}</b>: {detail}" if detail else f"• <b>{title}</b>"
            )
            continue

        title = next((cell for cell in padded_row if cell), "Row")
        rendered.append(f"• <b>{title}</b>")
        for header, cell in zip(headers, padded_row):
            if cell and cell != title:
                rendered.append(f"  {header}: {cell}")

    return rendered


_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]\n]*)\]\(([^)\n]+)\)")
_MARKDOWN_FILE_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")


def strip_telegram_image_refs(text: str) -> tuple[str, list[TelegramImageRef]]:
    """Remove local image refs from text and return refs Telegram should upload."""
    if not text:
        return "", []

    image_refs: list[TelegramImageRef] = []

    def replace_image(match: re.Match[str]) -> str:
        alt_text = match.group(1).strip()
        target = html.unescape(match.group(2).strip())
        file_id = _extract_local_file_id(target)
        if not file_id:
            return match.group(0)
        image_refs.append(TelegramImageRef(file_id=file_id, alt_text=alt_text))
        return ""

    cleaned = _MARKDOWN_IMAGE_RE.sub(replace_image, text)
    return _clean_stripped_markdown_refs(cleaned), image_refs


def strip_telegram_file_refs(text: str) -> tuple[str, list[TelegramFileRef]]:
    """Remove local file links from text and return refs Telegram should upload."""
    if not text:
        return "", []

    file_refs: list[TelegramFileRef] = []

    def replace_file_link(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        target = html.unescape(match.group(2).strip())
        file_id = _extract_local_file_id(target)
        if not file_id:
            return match.group(0)
        file_refs.append(TelegramFileRef(file_id=file_id, label=label))
        return ""

    cleaned = _MARKDOWN_FILE_LINK_RE.sub(replace_file_link, text)
    return _clean_stripped_markdown_refs(cleaned), file_refs


def _clean_stripped_markdown_refs(text: str) -> str:
    text = re.sub(r"(?m)^[ \t]*[-*•][ \t]*(?:\n|$)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def restore_telegram_task_context(
    agent_service: "AgentService",
    db: "Session",
    task_id: int,
) -> None:
    """Restore prior chat transcript and execution context for a Telegram turn."""
    from ...services.chat_history_service import load_task_transcript
    from ...services.task_execution_context_service import (
        load_task_execution_recovery_state,
    )

    agent_service.set_conversation_history(load_task_transcript(db, task_id))

    recovery_state: dict[str, Any] = await load_task_execution_recovery_state(
        db, task_id
    )
    agent_service.set_execution_context_messages(recovery_state.get("messages", []))
    agent_service.set_recovered_skill_context(recovery_state.get("skill_context"))


def persist_telegram_assistant_turn(
    db: "Session",
    task_id: int,
    user_id: int,
    content: str,
    interactions: list[dict[str, Any]] | None = None,
    message_type: str = "assistant_message",
) -> None:
    """Persist a Telegram assistant turn when it has text or structured prompts."""
    from ...services.chat_history_service import persist_assistant_message

    normalized_interactions = interactions or []
    if not content.strip() and not normalized_interactions:
        return

    persist_assistant_message(
        db=db,
        task_id=task_id,
        user_id=user_id,
        content=content,
        interactions=normalized_interactions,
        message_type=message_type,
    )


def _extract_local_file_id(target: str) -> str | None:
    internal_file_id = parse_file_id_ref(target)
    if internal_file_id is not None:
        return internal_file_id

    parsed = urlparse(target)
    path = parsed.path

    for prefix in ("/api/files/preview/", "/api/files/download/"):
        if path.startswith(prefix):
            return unquote(path[len(prefix) :].strip("/")) or None

    return None
