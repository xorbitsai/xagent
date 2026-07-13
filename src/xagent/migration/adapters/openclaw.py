"""Parse an OpenClaw (``~/.openclaw``) footprint into a migration bundle.

OpenClaw layout (see docs.openclaw.ai):

    ~/.openclaw/
        openclaw.json          JSON5 config (agents, models, channels, cron, ...)
        skills/                shared skills
        agents/ credentials/ sessions/
        workspace/
            SOUL.md IDENTITY.md AGENTS.md
            MEMORY.md USER.md memory/YYYY-MM-DD.md
            HEARTBEAT.md       natural-language schedule ("cron for your agent")
            skills/            workspace-scoped skills (highest precedence)
            .agents/skills/    project-shared skills

Personal cross-project skills also live in ``~/.agents/skills/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..bundle import (
    ArchivedItem,
    MigrationBundle,
    PersonaItem,
    ScheduleItem,
)
from .base import (
    SourceAdapter,
    collect_skill_dirs,
    load_skill_dir,
    normalize_cron_entries,
    parse_interval_seconds,
    read_text,
)

# Workspace docs that Hermes archives for manual review; we do the same rather
# than guess at a mapping. SOUL/IDENTITY are handled as persona instead.
_ARCHIVE_DOCS = ("TOOLS.md", "BOOTSTRAP.md", "AGENTS.md")


def _parse_jsonish(text: str) -> dict[str, Any] | None:
    """Parse JSON, tolerating the JSON5-isms OpenClaw permits.

    OpenClaw's ``openclaw.json`` is JSON5 (comments, trailing commas). We try
    strict JSON first, then strip those extensions and retry. Returns ``None``
    when the text cannot be parsed into a dict at all, so the caller can warn
    instead of silently migrating nothing.
    """
    text = text.strip()
    if not text:
        return {}
    for candidate in (text, _strip_json5(text)):
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return result if isinstance(result, dict) else None
    return None


def _strip_json5(text: str) -> str:
    """Drop // and /* */ comments and trailing commas, respecting strings.

    A naive regex strip would also eat ``//`` inside string values (e.g. a URL
    in the agent name) and corrupt the document, so this walks the text with a
    quote-state flag and only touches content outside double-quoted strings.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and text[i + 1 : i + 2] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and text[i + 1 : i + 2] == "*":
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if ch in "}]":
            # Trailing comma: drop a comma emitted just before this bracket.
            j = len(out) - 1
            while j >= 0 and out[j] in " \t\r\n":
                j -= 1
            if j >= 0 and out[j] == ",":
                del out[j]
        out.append(ch)
        i += 1
    return "".join(out)


def _dig(config: dict[str, Any], *path: str) -> Any:
    """Walk nested dict keys, returning ``None`` if any hop is missing."""
    node: Any = config
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


class OpenClawAdapter(SourceAdapter):
    key = "openclaw"

    def default_root(self) -> Path:
        return Path.home() / ".openclaw"

    def parse(self) -> MigrationBundle:
        root = self.root
        workspace = root / "workspace"
        bundle = MigrationBundle(source=self.key, source_root=str(root))

        config = _parse_jsonish(read_text(root / "openclaw.json"))
        if config is None:
            config = {}
            bundle.warnings.append(
                "Could not parse openclaw.json; the agent name and cron jobs "
                "from it were skipped."
            )

        bundle.agent_name = self._agent_name(config)
        bundle.persona = self._persona(workspace)
        bundle.skills = self._skills(root, workspace)
        bundle.schedules = self._schedules(config, workspace, bundle)
        self._archive_docs(workspace, bundle)
        return bundle

    def _agent_name(self, config: dict[str, Any]) -> str:
        name = _dig(config, "agents", "defaults", "name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return "OpenClaw Agent"

    def _persona(self, workspace: Path) -> PersonaItem | None:
        parts: list[str] = []
        for filename in ("SOUL.md", "IDENTITY.md"):
            text = read_text(workspace / filename).strip()
            if text:
                parts.append(text)
        if not parts:
            return None
        return PersonaItem(instructions="\n\n".join(parts))

    def _skills(self, root: Path, workspace: Path) -> list:
        # Highest precedence first, matching OpenClaw's own ordering.
        dirs = collect_skill_dirs(
            workspace / "skills",
            workspace / ".agents" / "skills",
            root / "skills",
            Path.home() / ".agents" / "skills",
        )
        skills = []
        for name, skill_dir in sorted(dirs.items()):
            item = load_skill_dir(name, skill_dir)
            if item is not None:
                item.slug = name
                skills.append(item)
        return skills

    def _schedules(
        self, config: dict[str, Any], workspace: Path, bundle: MigrationBundle
    ) -> list[ScheduleItem]:
        schedules: list[ScheduleItem] = []

        # 1. Structured cron entries in openclaw.json.
        entries = normalize_cron_entries(config.get("cron"))
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            prompt = entry.get("prompt") or entry.get("task") or ""
            expr = entry.get("schedule") or entry.get("cron") or entry.get("expression")
            schedules.append(
                ScheduleItem(
                    name=str(entry.get("name") or f"openclaw-cron-{index + 1}"),
                    prompt=str(prompt),
                    cron_expression=str(expr) if expr else None,
                    interval_seconds=parse_interval_seconds(entry),
                    source_path="openclaw.json:cron",
                )
            )

        # 2. HEARTBEAT.md -- natural-language schedule. We cannot reliably parse
        # free text into cron here, so each non-trivial line becomes a schedule
        # carrying its natural-language text for the loader to resolve.
        heartbeat = read_text(workspace / "HEARTBEAT.md").strip()
        if heartbeat:
            for index, line in enumerate(self._heartbeat_lines(heartbeat)):
                schedules.append(
                    ScheduleItem(
                        name=f"heartbeat-{index + 1}",
                        prompt=line,
                        natural_language=line,
                        source_path=str(workspace / "HEARTBEAT.md"),
                    )
                )
        return schedules

    @staticmethod
    def _heartbeat_lines(heartbeat: str) -> list[str]:
        """Extract actionable lines from HEARTBEAT.md, dropping headings/blanks."""
        lines: list[str] = []
        for raw in heartbeat.splitlines():
            line = raw.strip().lstrip("-*").strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
        return lines

    def _archive_docs(self, workspace: Path, bundle: MigrationBundle) -> None:
        for filename in _ARCHIVE_DOCS:
            path = workspace / filename
            if path.is_file():
                bundle.archived.append(
                    ArchivedItem(
                        name=filename,
                        reason=(
                            "No direct xagent equivalent; review and fold into "
                            "the agent instructions or a skill."
                        ),
                        content=path.read_bytes(),
                        source_path=str(path),
                    )
                )
