"""Load a :class:`MigrationBundle` into xagent, reusing existing services.

The loader is intentionally source-agnostic: it only consumes the neutral
bundle. Each artifact type maps to an existing xagent write path so migration
inherits the same validation and storage the product uses everywhere else:

* persona   -> an ``Agent`` (persona text becomes ``instructions``); re-runs
               reuse the agent created by an earlier migration instead of
               accumulating empty duplicates, and a bundle with nothing to
               attach (no persona, no importable schedule) creates none
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
    agent_reused: bool = False
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
        """Write the bundle into the database and report what happened.

        Consumes the bundle: schedules that cannot become triggers are appended
        to ``bundle.archived`` in place so that ``run_migration`` archives them
        to disk afterwards. Don't reuse or re-preview a loaded bundle.
        """
        report = LoadReport()
        agent = self._load_agent(bundle, report)
        self._load_skills(bundle, report)
        self._load_schedules(bundle, agent, report)
        self._record_archived(bundle, report)
        return report

    # -- agent / persona ---------------------------------------------------

    def _load_agent(self, bundle: MigrationBundle, report: LoadReport) -> Agent | None:
        """Return the agent that owns this import, creating it if needed.

        A bundle with no persona and no importable schedule gets no agent at
        all: skills are user-scoped, so an empty agent would just be clutter.
        Re-runs reuse the agent an earlier migration created (recognized by
        its description marker) rather than piling up duplicates that the
        user-level skill/trigger dedup would leave empty.
        """
        instructions = bundle.persona.instructions if bundle.persona else None
        has_importable_schedule = any(
            s.interval_seconds is not None for s in bundle.schedules
        )
        if instructions is None and not has_importable_schedule:
            return None

        existing = self._find_migrated_agent(bundle)
        if existing is not None:
            if instructions is not None:
                # Re-import syncs the persona from the source of truth.
                existing.instructions = instructions  # type: ignore[assignment]
                self.db.commit()
            report.agent_name = str(existing.name)
            report.agent_reused = True
            return existing

        name = self._unique_agent_name(bundle.agent_name)
        agent = Agent(
            user_id=self.user_id,
            name=name,
            # Doubles as the marker _find_migrated_agent keys re-run reuse on.
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

    def _find_migrated_agent(self, bundle: MigrationBundle) -> Agent | None:
        """Locate the agent a previous run of this migration created, if any.

        Matches on the description marker plus the (possibly uniquified) name,
        so a user's own same-named agent is never hijacked, while an agent an
        earlier run had to rename to "Name (2)" is still found.
        """
        base = _base_agent_name(bundle.agent_name)
        candidates = (
            self.db.query(Agent)
            .filter(
                Agent.user_id == self.user_id,
                Agent.description == f"Imported from {bundle.source}.",
            )
            .order_by(Agent.id.desc())
            .all()
        )
        for candidate in candidates:
            name = str(candidate.name)
            if name == base or name.startswith(f"{base} ("):
                return candidate
        return None

    def _unique_agent_name(self, desired: str) -> str:
        base = _base_agent_name(desired)
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
        # Committed name -> source name for this run, so a skip can say whether
        # it hit a pre-existing skill or a run-mate that normalized to the same
        # target (e.g. "my skill" vs "my-skill").
        imported_targets: dict[str, str] = {}
        for skill in bundle.skills:
            try:
                target_name = _normalize_skill_name(skill.name)
                imported_name = self._write_skill(
                    target_name=target_name,
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
                if target_name in imported_targets:
                    reason = (
                        "name collides with "
                        f"{imported_targets[target_name]!r} after normalization"
                    )
                else:
                    reason = "already exists"
                report.skills_skipped.append(f"{skill.name} ({reason})")
            else:
                report.skills_imported.append(imported_name)
                imported_targets[imported_name] = skill.name

    def _write_skill(
        self,
        *,
        target_name: str,
        files: dict[str, bytes],
        slug: str | None,
    ) -> str | None:
        """Insert one personal skill, honoring the conflict strategy.

        ``target_name`` is the already-normalized Skill Hub name. The actual
        write is delegated to Skill Hub's ``_write_personal_skill`` so
        migration inherits its name validation, path-traversal checks and
        total-size budget. Returns the stored name, or ``None`` when a
        conflict caused a skip.
        """
        from ..web.api.skill_hub import _write_personal_skill

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
        self, bundle: MigrationBundle, agent: Agent | None, report: LoadReport
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
                # Deliberate in-place mutation: write_archive(bundle) runs
                # after load() and picks these up from bundle.archived.
                bundle.archived.append(
                    ArchivedItem(
                        name=schedule.name,
                        reason=reason,
                        content=(schedule.prompt or "").encode("utf-8"),
                        source_path=schedule.source_path,
                    )
                )
                continue
            if agent is None:
                # _load_agent creates an agent whenever an importable schedule
                # exists; this only guards against future bundle changes.
                report.errors.append(
                    f"schedule {schedule.name!r}: no agent to attach the trigger to"
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


def _base_agent_name(desired: str) -> str:
    return (desired or "Imported Agent").strip()[:180]


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
        "agent_reused": report.agent_reused,
        "skills_imported": report.skills_imported,
        "skills_skipped": report.skills_skipped,
        "schedules_imported": report.schedules_imported,
        "schedules_skipped": report.schedules_skipped,
        "archived": report.archived,
        "errors": report.errors,
        "ok": report.ok,
    }
