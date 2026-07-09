"""Parse a Hermes (``~/.hermes``) footprint into a migration bundle.

Hermes layout (see hermes-agent.nousresearch.com/docs):

    ~/.hermes/
        config.yaml            model, agent, tts, mcp_servers, ...
        SOUL.md                persona
        memories/              MEMORY.md USER.md
        skills/                skill dirs (agentskills.io standard SKILL.md)
        cron/jobs.json         scheduled jobs (5-field cron expression + tz)

Hermes cron jobs are the one thing Hermes itself cannot migrate from OpenClaw
(it archives them). We parse them here so xagent can import them directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..bundle import MigrationBundle, PersonaItem, ScheduleItem
from .base import SourceAdapter, collect_skill_dirs, load_skill_dir, read_text


class HermesAdapter(SourceAdapter):
    key = "hermes"

    def default_root(self) -> Path:
        return Path.home() / ".hermes"

    def parse(self) -> MigrationBundle:
        root = self.root
        bundle = MigrationBundle(source=self.key, source_root=str(root))

        bundle.agent_name = "Hermes Agent"
        bundle.persona = self._persona(root)
        bundle.skills = self._skills(root)
        bundle.schedules = self._schedules(root, bundle)
        return bundle

    def _persona(self, root: Path) -> PersonaItem | None:
        path = root / "SOUL.md"
        text = read_text(path).strip()
        if not text:
            return None
        return PersonaItem(instructions=text, source_paths=[str(path)])

    def _skills(self, root: Path) -> list:
        dirs = collect_skill_dirs(root / "skills")
        skills = []
        for name, skill_dir in sorted(dirs.items()):
            item = load_skill_dir(name, skill_dir)
            if item is not None:
                item.slug = name
                skills.append(item)
        return skills

    def _schedules(self, root: Path, bundle: MigrationBundle) -> list[ScheduleItem]:
        jobs_path = root / "cron" / "jobs.json"
        text = read_text(jobs_path).strip()
        if not text:
            return []
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError:
            bundle.warnings.append(f"Could not parse {jobs_path}; skipping cron jobs.")
            return []

        if isinstance(data, dict):
            jobs = data.get("jobs")
            entries = jobs if isinstance(jobs, list) else list(data.values())
        elif isinstance(data, list):
            entries = data
        else:
            entries = []

        schedules: list[ScheduleItem] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            expr = entry.get("schedule") or entry.get("cron")
            schedules.append(
                ScheduleItem(
                    name=str(entry.get("name") or f"hermes-cron-{index + 1}"),
                    prompt=str(entry.get("prompt") or ""),
                    cron_expression=str(expr) if expr else None,
                    interval_seconds=self._interval(entry),
                    timezone=str(entry["timezone"]) if entry.get("timezone") else None,
                    deliver=str(entry["deliver"]) if entry.get("deliver") else None,
                    source_path=str(jobs_path),
                )
            )
        return schedules

    @staticmethod
    def _interval(entry: dict[str, Any]) -> int | None:
        value = entry.get("interval_seconds") or entry.get("intervalSeconds")
        if isinstance(value, int) and value > 0:
            return value
        return None
