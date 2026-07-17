"""Orchestrate a migration: detect -> parse -> preview -> load -> archive.

This is the engine behind ``xagent migrate``. It is deliberately free of
argparse/stdio wiring (that lives in the CLI) so it can be unit-tested and
reused by a future web endpoint.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from .adapters import detect_sources, get_adapter
from .adapters.base import SourceAdapter
from .bundle import MigrationBundle

if TYPE_CHECKING:
    from ..web.models.user import User
    from .loaders import LoadReport

logger = logging.getLogger(__name__)


def build_preview(bundle: MigrationBundle) -> dict[str, object]:
    """Summarize a bundle for the pre-migration confirmation screen."""
    interval_schedules = [s for s in bundle.schedules if s.interval_seconds is not None]
    cron_only = [s for s in bundle.schedules if s.interval_seconds is None]
    return {
        "source": bundle.source,
        "source_root": bundle.source_root,
        "agent_name": bundle.agent_name,
        "has_persona": bundle.persona is not None,
        "skills": [s.name for s in bundle.skills],
        "schedules_importable": [s.name for s in interval_schedules],
        "schedules_archived": [s.name for s in cron_only],
        "archived": [a.name for a in bundle.archived],
        "warnings": list(bundle.warnings),
    }


def resolve_adapters(
    source: str | None, source_dir: Path | None
) -> list[SourceAdapter]:
    """Pick adapters from an explicit ``--from`` or by auto-detection."""
    if source:
        return [get_adapter(source, root=source_dir)]
    if source_dir is not None:
        raise ValueError("--source-dir requires --from to name the platform.")
    return detect_sources()


def archive_dir_for(source: str, timestamp: datetime) -> Path:
    """Return the archive directory for a run, matching Hermes' layout."""
    from ..config import get_storage_root

    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return get_storage_root() / "migration" / source / stamp / "archive"


def write_archive(bundle: MigrationBundle, archive_dir: Path) -> list[str]:
    """Persist non-migratable items to disk for manual review.

    Returns the list of written file paths. Each item is written under a
    filesystem-safe name; a sibling ``REASON.txt`` records why it could not be
    migrated automatically.
    """
    if not bundle.archived:
        return []
    # From here on the DB import has already committed; a full disk, bad
    # permissions or a file squatting on the archive path must degrade to a
    # warning, not abort the run after the fact.
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "Could not create archive directory %s (%s); "
            "%d non-migratable item(s) were not written to disk.",
            archive_dir,
            exc,
            len(bundle.archived),
        )
        return []
    written: list[str] = []
    reasons: list[str] = []
    for index, item in enumerate(bundle.archived):
        safe = _safe_name(item.name) or f"item-{index + 1}"
        out_path = archive_dir / safe
        # Avoid clobbering same-named archived items (e.g. two "TOOLS.md"),
        # re-checking each fallback since item names may themselves collide
        # with a generated "N-name".
        counter = 2
        while out_path.exists():
            out_path = archive_dir / f"{counter}-{safe}"
            counter += 1
        try:
            out_path.write_bytes(item.content or b"")
            written.append(str(out_path))
        except OSError:
            continue
        reasons.append(f"{out_path.name}\t<- {item.source_path}\n  {item.reason}")
    if reasons:
        try:
            (archive_dir / "REASON.txt").write_text(
                "\n".join(reasons) + "\n", encoding="utf-8"
            )
        except OSError:
            pass
    return written


def _safe_name(name: str) -> str:
    keep = [c if c.isalnum() or c in "-._" else "_" for c in name]
    return "".join(keep).strip("._")[:120]


def run_migration(
    db: Session,
    *,
    user: "User",
    bundle: MigrationBundle,
    skill_conflict: str = "skip",
    now: datetime | None = None,
) -> tuple["LoadReport", list[str]]:
    """Load a bundle and archive its non-migratable items.

    Returns the load report and the list of archived file paths.
    """
    # Imported lazily so the parse/preview path (used by ``--dry-run`` and the
    # confirmation screen) does not pull in the web DB model stack.
    from .loaders import MigrationLoader

    loader = MigrationLoader(db, user=user, skill_conflict=skill_conflict)
    report = loader.load(bundle)
    stamp = now or datetime.now(timezone.utc)
    archive_paths = write_archive(bundle, archive_dir_for(bundle.source, stamp))
    return report, archive_paths
