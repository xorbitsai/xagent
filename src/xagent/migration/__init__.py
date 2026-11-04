"""Migrate customization footprints from other agent platforms into xagent.

This package implements ``xagent migrate``, the counterpart to Hermes'
``hermes claw migrate``. It reads a source agent platform's on-disk state
(OpenClaw's ``~/.openclaw`` or Hermes' ``~/.hermes``), normalizes it into a
platform-neutral :class:`~xagent.migration.bundle.MigrationBundle`, previews
what will be imported, and then loads it into xagent.

Where Hermes archives cron/heartbeat jobs for manual re-creation, xagent
imports them directly as scheduled ``AgentTrigger`` rows -- see
``xagent.migration.loaders``.
"""

from .bundle import (
    ArchivedItem,
    MigrationBundle,
    PersonaItem,
    ScheduleItem,
    SkillItem,
)

__all__ = [
    "ArchivedItem",
    "MigrationBundle",
    "PersonaItem",
    "ScheduleItem",
    "SkillItem",
]
