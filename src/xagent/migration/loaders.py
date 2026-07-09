"""Load a :class:`MigrationBundle` into xagent, reusing existing services.

The loader is intentionally source-agnostic: it only consumes the neutral
bundle. Each artifact type maps to an existing xagent write path so migration
inherits the same validation and storage the product uses everywhere else:

* persona   -> a new ``Agent`` (persona text becomes ``instructions``)
* skills    -> personal ``UserSkill`` rows (same writer as Skill Hub imports)
* schedules -> scheduled ``AgentTrigger`` rows when an interval is known;
               cron-expression jobs are archived pending the cron engine
               (see :data:`CRON_UNSUPPORTED_REASON`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from ..web.models.agent import Agent
from ..web.models.skill import UserSkill, UserSkillFile
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


@dataclass
class LoadReport:
    """Per-run tally of what happened, mirroring Hermes' migration report."""

    agent_name: str | None = None
    skills_imported: list[str] = field(default_factory=list)
    skills_skipped: list[str] = field(default_factory=list)
    schedules_imported: list[str] = field(default_factory=list)
    archived: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


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
                    version=skill.version,
                )
            except Exception as exc:  # pragma: no cover - defensive per-skill
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
        version: str | None,
    ) -> str | None:
        """Insert one personal skill, honoring the conflict strategy.

        Returns the stored name, or ``None`` when a conflict caused a skip.
        """
        existing = (
            self.db.query(UserSkill)
            .filter(UserSkill.user_id == self.user_id, UserSkill.name == name)
            .first()
        )
        target_name = name
        if existing is not None:
            if self.skill_conflict == "skip":
                return None
            if self.skill_conflict == "overwrite":
                self.db.delete(existing)
                self.db.flush()
            elif self.skill_conflict == "rename":
                target_name = self._unique_skill_name(name)

        skill_row = UserSkill(
            user_id=self.user_id,
            name=target_name,
            origin="imported",
            clawhub_slug=slug,
            clawhub_version=version,
            created_by_user_id=self.user_id,
            updated_by_user_id=self.user_id,
        )
        self.db.add(skill_row)
        self.db.flush()
        for path, content in sorted(files.items()):
            self.db.add(
                UserSkillFile(
                    skill_id=skill_row.id,
                    path=path,
                    content=content,
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                    media_type=_guess_media_type(path),
                )
            )
        self.db.commit()
        return target_name

    def _unique_skill_name(self, desired: str) -> str:
        candidate = f"{desired}-imported"
        suffix = 2
        while (
            self.db.query(UserSkill.id)
            .filter(UserSkill.user_id == self.user_id, UserSkill.name == candidate)
            .first()
            is not None
        ):
            candidate = f"{desired}-imported-{suffix}"
            suffix += 1
        return candidate

    # -- schedules ---------------------------------------------------------

    def _load_schedules(
        self, bundle: MigrationBundle, agent: Agent, report: LoadReport
    ) -> None:
        from ..web.services.triggers import create_agent_trigger

        for schedule in bundle.schedules:
            # The scheduler only understands intervals today. A job that only
            # carries a cron expression is archived rather than mis-scheduled.
            if schedule.interval_seconds is None:
                bundle.archived.append(
                    ArchivedItem(
                        name=schedule.name,
                        reason=CRON_UNSUPPORTED_REASON,
                        content=(schedule.prompt or "").encode("utf-8"),
                        source_path=schedule.source_path,
                    )
                )
                continue
            try:
                create_agent_trigger(
                    self.db,
                    user_id=self.user_id,
                    agent_id=int(agent.id),
                    trigger_type="scheduled",
                    name=schedule.name[:200],
                    config={"interval_seconds": int(schedule.interval_seconds)},
                    prompt_template=schedule.prompt or None,
                )
            except Exception as exc:  # pragma: no cover - defensive per-schedule
                report.errors.append(f"schedule {schedule.name!r}: {exc}")
                continue
            report.schedules_imported.append(schedule.name)

    # -- archive -----------------------------------------------------------

    def _record_archived(self, bundle: MigrationBundle, report: LoadReport) -> None:
        for item in bundle.archived:
            report.archived.append(item.name)


def _guess_media_type(path: str) -> str | None:
    """Reuse the Skill-library media-type guess, tolerating import failure."""
    try:
        from ..skills.library import guess_media_type

        return guess_media_type(path)
    except Exception:  # pragma: no cover - fallback only
        return None


def as_dict(report: LoadReport) -> dict[str, Any]:
    """Serialize a report for JSON output / logging."""
    return {
        "agent_name": report.agent_name,
        "skills_imported": report.skills_imported,
        "skills_skipped": report.skills_skipped,
        "schedules_imported": report.schedules_imported,
        "archived": report.archived,
        "errors": report.errors,
    }
