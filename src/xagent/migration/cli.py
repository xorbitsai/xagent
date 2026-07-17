"""``xagent migrate`` command line interface.

Counterpart to ``hermes claw migrate``: auto-detects a source platform on this
machine (or takes ``--from``/``--source-dir``), previews what will be imported,
asks for confirmation, then loads it and reports what happened -- archiving
anything that could not be migrated automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from .adapters.base import SourceAdapter
from .bundle import MigrationBundle
from .runner import build_preview, resolve_adapters, run_migration

if TYPE_CHECKING:
    from ..web.models.user import User
    from .loaders import LoadReport


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--from",
        dest="source",
        choices=["openclaw", "hermes"],
        help="Source platform. Omit to auto-detect installed platforms.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Override the source home directory (requires --from).",
    )
    parser.add_argument(
        "--user",
        dest="username",
        help="Target xagent username. Defaults to the sole/admin user.",
    )
    parser.add_argument(
        "--skill-conflict",
        choices=["skip", "overwrite", "rename"],
        default="skip",
        help="What to do when a skill of the same name already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the preview and exit without importing anything.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )


def _resolve_user(db: Session, username: str | None) -> "User":
    from ..web.models.user import User

    if username:
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            raise SystemExit(f"No xagent user named {username!r}.")
        return user

    users = db.query(User).order_by(User.id).all()
    if not users:
        raise SystemExit(
            "No xagent users exist yet. Start the web app and create the first "
            "admin user, then re-run migrate."
        )
    admins = [u for u in users if bool(u.is_admin)]
    if len(users) == 1:
        return users[0]
    if len(admins) == 1:
        return admins[0]
    names = ", ".join(str(u.username) for u in users)
    raise SystemExit(
        f"Multiple users exist ({names}); pass --user to choose the target."
    )


def _print_preview(preview: dict) -> None:
    print(f"\nSource: {preview['source']}  ({preview['source_root']})")
    print(f"Target agent: {preview['agent_name']}")
    print(f"  persona/instructions : {'yes' if preview['has_persona'] else 'no'}")
    print(f"  skills               : {len(preview['skills'])}")
    for name in preview["skills"]:
        print(f"      - {name}")
    print(f"  scheduled triggers   : {len(preview['schedules_importable'])}")
    for name in preview["schedules_importable"]:
        print(f"      - {name}")
    archived = list(preview["schedules_archived"]) + list(preview["archived"])
    if archived:
        print(f"  archived (manual)    : {len(archived)}")
        for name in archived:
            print(f"      - {name}")
    for warning in preview["warnings"]:
        print(f"  ! {warning}")


def _print_report(report: "LoadReport", archive_paths: list[str]) -> None:
    if report.ok:
        print("\nMigration complete.")
    else:
        print("\nMigration finished with errors; some items were not imported.")
    if report.agent_name is None:
        print("  agent                : none needed (skills only)")
    elif report.agent_reused:
        print(f"  agent reused         : {report.agent_name}")
    else:
        print(f"  agent created        : {report.agent_name}")
    print(f"  skills imported      : {len(report.skills_imported)}")
    if report.skills_skipped:
        # Each entry carries its reason ("already exists" vs a name collision
        # within this import), so print them rather than a bare count.
        print(f"  skills skipped       : {len(report.skills_skipped)}")
        for name in report.skills_skipped:
            print(f"      - {name}")
    print(f"  schedules imported   : {len(report.schedules_imported)}")
    if report.schedules_skipped:
        print(
            f"  schedules skipped    : {len(report.schedules_skipped)}"
            " (already imported)"
        )
    if report.archived:
        print(f"  archived (manual)    : {len(report.archived)}")
    if archive_paths:
        archive_root = str(Path(archive_paths[0]).parent)
        print(f"  archive directory    : {archive_root}")
    for error in report.errors:
        print(f"  ! {error}")


def _parse_and_preview(adapter: SourceAdapter) -> MigrationBundle:
    bundle = adapter.parse()
    _print_preview(build_preview(bundle))
    return bundle


def run(args: argparse.Namespace) -> int:
    try:
        adapters = resolve_adapters(args.source, args.source_dir)
    except ValueError as exc:
        raise SystemExit(str(exc))

    if not adapters:
        print(
            "No source platform detected. Looked for ~/.openclaw and ~/.hermes.\n"
            "Pass --from openclaw|hermes with --source-dir to point at a copy."
        )
        return 1

    if args.dry_run:
        # Parse/preview only. The DB stack is never initialized, so a fresh
        # install (no users yet) can still preview what a migration would do.
        for adapter in adapters:
            bundle = _parse_and_preview(adapter)
            if bundle.is_empty():
                print("  (nothing to import from this source)")
            else:
                print("  dry-run: no changes made.")
        return 0

    from ..web.models.database import get_session_local, init_db

    init_db()
    session_local = get_session_local()
    db: Session = session_local()
    try:
        user = _resolve_user(db, args.username)
        print(f"Importing into xagent user: {user.username}")

        exit_code = 0
        for adapter in adapters:
            bundle = _parse_and_preview(adapter)
            if bundle.is_empty():
                print("  (nothing to import from this source)")
                continue
            if not args.yes and not _confirm(str(user.username)):
                print("  skipped.")
                continue
            report, archive_paths = run_migration(
                db,
                user=user,
                bundle=bundle,
                skill_conflict=args.skill_conflict,
            )
            _print_report(report, archive_paths)
            if report.errors:
                exit_code = 1
        return exit_code
    finally:
        db.close()


def _confirm(username: str) -> bool:
    # Restate the target account: _resolve_user may have auto-selected it, and
    # a wrong target should be catchable at the moment of confirmation.
    try:
        answer = (
            input(f"\nImport into xagent user {username!r}? [y/N] ").strip().lower()
        )
    except EOFError:
        return False
    return answer in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xagent migrate",
        description="Import an agent from OpenClaw or Hermes into xagent.",
    )
    add_arguments(parser)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return run(args)
