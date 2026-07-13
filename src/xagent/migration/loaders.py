"""Load a :class:`MigrationBundle` into xagent, reusing existing services.

The loader is intentionally source-agnostic: it only consumes the neutral
bundle. Each artifact type maps to an existing xagent write path so migration
inherits the same validation and storage the product uses everywhere else:

* persona   -> a new ``Agent`` (persona text becomes ``instructions``)
* skills    -> personal ``UserSkill`` rows (same writer as Skill Hub imports)
* schedules -> scheduled ``AgentTrigger`` rows when an interval is known;
               cron-expression jobs and natural-language heartbeat lines are
               archived pending the cron engine (see
               :data:`CRON_UNSUPPORTED_REASON` /
               :data:`HEARTBEAT_UNSUPPORTED_REASON`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from ..web.models.agent import Agent
from ..web.models.skill import UserSkill
from ..web.models.user import User
from .bundle import ArchivedItem, MigrationBundle

# xagent's scheduled triggers currently fire on a fixed interval only; standard
# 5-field cron expressions are not yet supported by the scheduler. Until the
# cron engine lands (planned follow-up), such jobs are archived with this reason
# rather than silently approximated into the wrong cadence.
CRON_UNSUPPORTED_REASON = (
    "Cron-expression schedules are not yet supported by the xagent scheduler; "
    "recreate this job as an interval-based trigger, or wait for cron support."
)

# HEARTBEAT.md lines carry a schedule in free text ("Check HN each morning").
# We cannot reliably turn that into an interval, so they are archived with a
# reason that says so instead of the cron-expression one.
HEARTBEAT_UNSUPPORTED_REASON = (
    "Natural-language schedules (HEARTBEAT.md) cannot be translated into a "
    "trigger automatically; recreate this line as an interval-based trigger."
)

# Skill Hub requires names matching [A-Za-z0-9_-]+; source directory names may
# contain anything the filesystem allows, so runs of other characters collapse
# to a single dash before insert.
_INVALID_SKILL_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass
class LoadReport:
    """Per-run tally of what happened, mirroring Hermes' migration report."""

    agent_name: str | None = None
    skills_imported: list[str] = field(default_factory=list)
    skills_skipped: list[str] = field(default_factory=list)
    schedules_imported: list[str] = field(default_factory=list)
    schedules_skipped: list[str] = field(default_factory=list)
    archived: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every artifact loaded cleanly (a partial import is not ok)."""
        return not self.errors


class MigrationLoader:
    """Write a parsed bundle into the database for a target user."""

    def __init__(
        self,
        db: Session,
        *,
        user: User,
        skill_conflict: str = "skip",
    ) -> None:
        self.db = db
        self.user = user
        self.user_id = int(user.id)
        if skill_conflict not in {"skip", "overwrite", "rename"}:
            raise ValueError(f"Unknown skill_conflict strategy {skill_conflict!r}")
        self.skill_conflict = skill_conflict

    def load(self, bundle: MigrationBundle) -> LoadReport:
        report = LoadReport()
        agent = self._load_agent(bundle, report)
        self._load_skills(bundle, report)
        self._load_schedules(bundle, agent, report)
        self._record_archived(bundle, report)
        return report

    # -- agent / persona ---------------------------------------------------

    def _load_agent(self, bundle: MigrationBundle, report: LoadReport) -> Agent:
        """Create the owning agent, carrying persona text into instructions."""
        instructions = bundle.persona.instructions if bundle.persona else None
        name = self._unique_agent_name(bundle.agent_name)
        agent = Agent(
            user_id=self.user_id,
            name=name,
            description=f"Imported from {bundle.source}.",
            instructions=instructions,
            execution_mode="balanced",
            models={},
            knowledge_bases=[],
            skills=[],
            tool_categories=[],
            suggested_prompts=[],
        )
        self.db.add(agent)
        self.db.commit()
        self.db.refresh(agent)
        report.agent_name = name
        return agent

    def _unique_agent_name(self, desired: str) -> str:
        base = (desired or "Imported Agent").strip()[:180]
        candidate = base
        suffix = 2
        while (
            self.db.query(Agent.id)
            .filter(Agent.user_id == self.user_id, Agent.name == candidate)
            .first()
            is not None
        ):
            candidate = f"{base} ({suffix})"
            suffix += 1
        return candidate

    # -- skills ------------------------------------------------------------

    def _load_skills(self, bundle: MigrationBundle, report: LoadReport) -> None:
        for skill in bundle.skills:
            try:
                imported_name = self._write_skill(
                    name=skill.name,
                    files=skill.files,
                    slug=skill.slug,
                )
            except Exception as exc:
                # Reset the session so one bad skill cannot poison every
                # subsequent write in this run with PendingRollbackError.
                self.db.rollback()
                report.errors.append(f"skill {skill.name!r}: {exc}")
                continue
            if imported_name is None:
                report.skills_skipped.append(skill.name)
            else:
                report.skills_imported.append(imported_name)

    def _write_skill(
        self,
        *,
        name: str,
        files: dict[str, bytes],
        slug: str | None,
    ) -> str | None:
        """Insert one personal skill, honoring the conflict strategy.

        The actual write is delegated to Skill Hub's ``_write_personal_skill``
        so migration inherits its name validation, path-traversal checks and
        total-size budget. Returns the stored name, or ``None`` when a
        conflict caused a skip.
        """
        from ..web.api.skill_hub import _write_personal_skill

        target_name = _normalize_skill_name(name)
        existing = (
            self.db.query(UserSkill)
            .filter(UserSkill.user_id == self.user_id, UserSkill.name == target_name)
            .first()
        )
        if existing is not None:
            if self.skill_conflict == "skip":
                return None
            if self.skill_conflict == "overwrite":
                self.db.delete(existing)
                self.db.flush()
            elif self.skill_conflict == "rename":
                target_name = self._unique_skill_name(target_name)

        _write_personal_skill(
            db=self.db,
            user=self.user,
            name=target_name,
            files=files,
            origin="imported",
            clawhub_slug=slug[:128] if slug else None,
        )
        return target_name

    def _unique_skill_name(self, desired: str) -> str:
        # Leave room for the suffix inside UserSkill.name's 100-char column.
        base = desired[:80]
        candidate = f"{base}-imported"
        suffix = 2
        while (
            self.db.query(UserSkill.id)
            .filter(UserSkill.user_id == self.user_id, UserSkill.name == candidate)
            .first()
            is not None
        ):
            candidate = f"{base}-imported-{suffix}"
            suffix += 1
        return candidate

    # -- schedules ---------------------------------------------------------

    def _load_schedules(
        self, bundle: MigrationBundle, agent: Agent, report: LoadReport
    ) -> None:
        from ..web.services.triggers import create_agent_trigger

        for schedule in bundle.schedules:
            # The scheduler only understands intervals today. Cron-expression
            # jobs and natural-language heartbeat lines are archived (each with
            # its own reason) rather than mis-scheduled.
            if schedule.interval_seconds is None:
                reason = (
                    HEARTBEAT_UNSUPPORTED_REASON
                    if schedule.natural_language
                    else CRON_UNSUPPORTED_REASON
                )
                bundle.archived.append(
                    ArchivedItem(
                        name=schedule.name,
                        reason=reason,
                        content=(schedule.prompt or "").encode("utf-8"),
                        source_path=schedule.source_path,
                    )
                )
                continue
            trigger_name = schedule.name[:200]
            interval = int(schedule.interval_seconds)
            prompt = schedule.prompt or None
            if self._trigger_exists(
                name=trigger_name, interval=interval, prompt=prompt
            ):
                report.schedules_skipped.append(schedule.name)
                continue
            try:
                create_agent_trigger(
                    self.db,
                    user_id=self.user_id,
                    agent_id=int(agent.id),
                    trigger_type="scheduled",
                    name=trigger_name,
                    config={"interval_seconds": interval},
                    prompt_template=prompt,
                )
            except Exception as exc:
                # Same session hygiene as the skill path above.
                self.db.rollback()
                report.errors.append(f"schedule {schedule.name!r}: {exc}")
                continue
            report.schedules_imported.append(schedule.name)

    def _trigger_exists(self, *, name: str, interval: int, prompt: str | None) -> bool:
        """True when an earlier migration run already created this trigger.

        Agents are renamed per run, so the lookup goes by the user's triggers
        rather than the agent's -- otherwise every re-run would add another
        independently-firing copy of the same source job.
        """
        from ..web.models.trigger import AgentTrigger

        candidates = (
            self.db.query(AgentTrigger)
            .filter(
                AgentTrigger.user_id == self.user_id,
                AgentTrigger.type == "scheduled",
                AgentTrigger.name == name,
            )
            .all()
        )
        for candidate in candidates:
            raw_config = candidate.config
            config: dict[str, Any] = raw_config if isinstance(raw_config, dict) else {}
            if (
                config.get("interval_seconds") == interval
                and (candidate.prompt_template or None) == prompt
            ):
                return True
        return False

    # -- archive -----------------------------------------------------------

    def _record_archived(self, bundle: MigrationBundle, report: LoadReport) -> None:
        for item in bundle.archived:
            report.archived.append(item.name)


def _normalize_skill_name(name: str) -> str:
    """Map a source directory name onto Skill Hub's naming rule."""
    cleaned = _INVALID_SKILL_NAME_CHARS.sub("-", name).strip("-_")[:100]
    if not cleaned:
        raise ValueError(f"skill name {name!r} has no usable characters")
    return cleaned


def as_dict(report: LoadReport) -> dict[str, Any]:
    """Serialize a report for JSON output / logging."""
    return {
        "agent_name": report.agent_name,
        "skills_imported": report.skills_imported,
        "skills_skipped": report.skills_skipped,
        "schedules_imported": report.schedules_imported,
        "schedules_skipped": report.schedules_skipped,
        "archived": report.archived,
        "errors": report.errors,
        "ok": report.ok,
    }
