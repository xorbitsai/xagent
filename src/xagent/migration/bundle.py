"""Platform-neutral intermediate representation for a migration.

Every source adapter (OpenClaw, Hermes) parses its own on-disk layout into a
single :class:`MigrationBundle`. Downstream planning, preview and loading only
ever look at the bundle, so the loader logic is written once regardless of
source. This mirrors the shape Hermes' migration reports use
(migrated / skipped / archived), so the UX carries over.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SkillItem:
    """A skill discovered in the source platform.

    ``files`` maps a relative path (always containing ``SKILL.md``) to its raw
    bytes, matching the shape :func:`xagent.skills.parser.SkillParser.parse_bundle`
    and the personal-skill writer expect.
    """

    name: str
    files: dict[str, bytes]
    source_path: str
    description: str = ""
    slug: str | None = None


@dataclass
class PersonaItem:
    """Persona / identity text destined for an agent's ``instructions``.

    OpenClaw splits this across ``SOUL.md`` (+ ``IDENTITY.md``); Hermes keeps a
    single ``SOUL.md``. Adapters concatenate the pieces into ``instructions``.
    """

    instructions: str


@dataclass
class ScheduleItem:
    """A recurring/one-shot job to become a scheduled ``AgentTrigger``.

    ``cron_expression`` is a standard 5-field cron string when the source used
    one; ``interval_seconds`` is the fallback when only an interval is known.
    ``natural_language`` carries an unparsed HEARTBEAT.md line; the loader
    cannot turn it into a trigger yet, so it archives such items with a
    heartbeat-specific reason for manual re-creation (see loaders).
    """

    name: str
    prompt: str
    cron_expression: str | None = None
    interval_seconds: int | None = None
    natural_language: str | None = None
    source_path: str = ""


@dataclass
class ArchivedItem:
    """Something we could not migrate automatically, kept for manual review.

    ``reason`` explains why (unsupported concept, missing target, ambiguous
    schedule, ...). ``content`` is written verbatim into the archive dir so the
    user still has it.
    """

    name: str
    reason: str
    content: bytes = b""
    source_path: str = ""


@dataclass
class MigrationBundle:
    """Everything a single source platform yields, normalized.

    ``source`` is the platform key (``"openclaw"`` / ``"hermes"``);
    ``source_root`` is the directory it was read from. ``warnings`` collects
    non-fatal parse issues to surface in the preview.
    """

    source: str
    source_root: str
    agent_name: str = "Imported Agent"
    persona: PersonaItem | None = None
    skills: list[SkillItem] = field(default_factory=list)
    schedules: list[ScheduleItem] = field(default_factory=list)
    archived: list[ArchivedItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.persona or self.skills or self.schedules or self.archived)
