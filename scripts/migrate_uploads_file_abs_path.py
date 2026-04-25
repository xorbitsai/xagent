#!/usr/bin/env python3
"""
Migrate uploaded_files.storage_path from absolute to relative paths.

This script converts existing absolute path records to relative paths.
Supports multiple upload directories for migration from previous locations.

By default, runs in dry-run mode to preview changes. Use --confirm to execute.

REQUIREMENTS:
    1. Run from the xagent project root directory (where .env is located)
    2. Use the same Python environment that runs xagent
    3. Database must be accessible (same configuration as xagent)

Commands:
    check      Check migration status (no parameters needed)
    migrate    Migrate absolute paths to relative paths (requires -d)

Usage:
    python scripts/migrate_uploads_file_abs_path.py check
    python scripts/migrate_uploads_file_abs_path.py migrate -d PATH [--confirm]

Options for migrate:
    --uploads-dir, -d  Uploads directory to convert (required, can be specified multiple times)
    --confirm          Actually modify database (default is dry-run mode)
    --verbose, -v      Show all records instead of just first 10
    --batch-size, -b   Batch size for streaming and commits (default: 1000)

Examples:
    # Check migration status
    python scripts/migrate_uploads_file_abs_path.py check

    # Preview changes (dry-run mode)
    python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path/uploads

    # Preview from multiple locations
    python scripts/migrate_uploads_file_abs_path.py migrate -d /old1/uploads -d /old2/uploads

    # Actually execute migration
    python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path/uploads --confirm

    # Show all records in preview
    python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path/uploads -v

    # Custom batch size
    python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path/uploads -b 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from sqlalchemy import func, select

if TYPE_CHECKING:
    from _typeshed import StrPath

# Setup logging (only show warnings and errors, use rich for user output)
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("migrate_uploads_file_abs_path")

console = Console()


def init_database():
    """Common database initialization for check and migrate commands.

    Returns:
        tuple: (SessionLocal factory, UploadedFile model class)
    """
    console.print("[cyan]Loading environment...[/cyan]")
    load_dotenv()

    console.print("[cyan]Importing xagent modules...[/cyan]")
    from xagent.web.models.database import get_engine, get_session_local, init_db
    from xagent.web.models.uploaded_file import UploadedFile

    console.print("[cyan]Initializing xagent database...[/cyan]")
    init_db()
    engine = get_engine()
    console.print(f"[cyan]Database: {engine.url}[/cyan]")

    return get_session_local, UploadedFile


def check() -> None:
    """Check migration status - count absolute vs relative paths in database."""
    session_maker, UploadedFile = init_database()
    SessionLocal = session_maker()

    console.print("\n[cyan]Checking migration status...[/cyan]\n")

    with SessionLocal() as session:
        total_count = (
            session.execute(select(func.count()).select_from(UploadedFile)).scalar()
            or 0
        )

        if total_count == 0:
            console.print("[yellow]No records found in database.[/yellow]")
            return

        console.print(f"[dim]Total records in database: {total_count}[/dim]\n")

        # Count absolute vs relative paths with progress bar
        absolute_count = 0
        relative_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            check_task = progress.add_task(
                "[cyan]Scanning records...", total=total_count
            )

            stmt = select(UploadedFile).execution_options(yield_per=1000)
            for record in session.execute(stmt).scalars():
                storage_str = str(record.storage_path)

                # Detect if this is a Windows-style absolute path
                is_win_style = len(storage_str) >= 2 and storage_str[1] == ":"
                if is_win_style:
                    storage_path = PureWindowsPath(storage_str)
                else:
                    storage_path = PurePosixPath(storage_str)

                if storage_path.is_absolute():
                    absolute_count += 1
                else:
                    relative_count += 1

                progress.update(check_task, advance=1)

    # Build summary table
    table = Table(
        title="Migration Status", show_header=True, header_style="bold magenta"
    )
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    pct_abs = f"{absolute_count / total_count * 100:.1f}%" if total_count > 0 else "0%"
    pct_rel = f"{relative_count / total_count * 100:.1f}%" if total_count > 0 else "0%"

    table.add_row("Total records", str(total_count), "")
    table.add_row("Absolute paths (need migration)", str(absolute_count), pct_abs)
    table.add_row("Relative paths (already migrated)", str(relative_count), pct_rel)

    console.print("\n")
    console.print(table)

    if absolute_count == 0:
        console.print(
            "\n[bold green]✓ Migration complete! No absolute paths found.[/bold green]"
        )
    else:
        console.print(
            f"\n[bold yellow]⚠ {absolute_count} absolute paths need migration.[/bold yellow]"
        )
        console.print(
            "[dim]Run: python scripts/migrate_uploads_file_abs_path.py migrate -d <uploads_dir>[/dim]"
        )


def migrate(
    uploads_dirs: list[StrPath], batch_size: int, confirm: bool, verbose: bool = False
) -> None:
    """Migrate absolute paths to relative paths in uploaded_files table."""
    # Resolve uploads directories
    console.print("[cyan]Uploads directories to convert:[/cyan]")
    for d in uploads_dirs:
        console.print(f"  - {d}")
    console.print(f"[dim]Batch size: {batch_size}[/dim]\n")

    console.print("[cyan]Loading environment...[/cyan]")
    load_dotenv(verbose=verbose)

    session_maker, UploadedFile = init_database()
    SessionLocal = session_maker()

    with SessionLocal() as session:
        # Get total count for progress bar
        total_count = session.execute(
            select(func.count()).select_from(UploadedFile)
        ).scalar()
        console.print(f"[dim]Total records in database: {total_count}[/dim]\n")

        if total_count == 0:
            console.print("[yellow]No records found in database.[/yellow]")
            return

        # Statistics
        migrated_count = 0
        skipped_relative = 0
        skipped_external = 0
        errors = 0

        # Sample records for display (first 10)
        sample_migrated = []

        console.print(
            f"\n[cyan]Phase 1: {'Scanning and migrating...' if confirm else 'Scanning (dry-run mode)...'}[/cyan]"
        )

        # Scan with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            scan_task = progress.add_task(
                "[cyan]Processing records...", total=total_count
            )

            stmt = select(UploadedFile).execution_options(yield_per=batch_size)
            for record in session.execute(stmt).scalars():
                storage_str = str(record.storage_path)

                # Detect if this is a Windows-style absolute path (has drive letter like C:)
                # Use the appropriate Path class to check if it's absolute
                is_win_style = len(storage_str) >= 2 and storage_str[1] == ":"
                if is_win_style:
                    storage_path = PureWindowsPath(storage_str)
                    is_abs = storage_path.is_absolute()
                else:
                    storage_path = PurePosixPath(storage_str)
                    is_abs = storage_path.is_absolute()

                # Skip if already relative
                if not is_abs:
                    skipped_relative += 1
                    progress.update(scan_task, advance=1)
                    continue

                # Try to match against any uploads root
                matched = False
                new_path = None
                for root_dir in uploads_dirs:
                    # Reuse the same Path class detection as above
                    # storage_path is already set as PureWindowsPath or PurePosixPath
                    user_root = (
                        storage_path.__class__(root_dir) / f"user_{record.user_id}"
                    )

                    try:
                        # Check if path is under this root
                        relative = storage_path.relative_to(user_root)
                        matched = True
                        new_path = relative.as_posix()
                        break
                    except ValueError:
                        # Path not under this root, try next
                        continue

                if not matched:
                    # Absolute path but outside any uploads_root
                    skipped_external += 1
                    progress.update(scan_task, advance=1)
                    continue

                # Collect sample for display
                old_path = str(record.storage_path)
                if len(sample_migrated) < 10:
                    sample_migrated.append(
                        {
                            "file_id": str(record.file_id),
                            "old_path": old_path,
                            "new_path": new_path,
                        }
                    )

                migrated_count += 1

                # Only modify database if not in dry-run mode
                if confirm:
                    record.storage_path = new_path  # type: ignore[assignment]

                    # Batch commit (progress bar shows current count)
                    if migrated_count % batch_size == 0:
                        session.commit()

                progress.update(scan_task, advance=1)

            # Final commit for remaining records
            if confirm and migrated_count % batch_size != 0:
                session.commit()

        # Build summary table
        table = Table(
            title="Migration Summary", show_header=True, header_style="bold magenta"
        )
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Status", justify="center")

        table.add_row("Total scanned", str(total_count), "")
        table.add_row(
            "Relative paths (skip)", str(skipped_relative), "[yellow]✓[/yellow]"
        )
        table.add_row(
            "Other absolute paths (skip)", str(skipped_external), "[yellow]✓[/yellow]"
        )
        table.add_row("To migrate", str(migrated_count), "[green]✓[/green]")
        if errors:
            table.add_row("Errors", str(errors), "[red]✗[/red]")

        console.print("\n")
        console.print(table)

        # Show hint for external paths
        if skipped_external > 0:
            console.print(
                f"\n[bold yellow]💡 {skipped_external} absolute paths don't match any --uploads-dir[/bold yellow]"
            )
            console.print(
                "[dim]Add more -d options to specify additional directories[/dim]"
            )

        # Show sample of migrated records
        if sample_migrated:
            console.print(
                f"\n[bold cyan]Preview[/bold cyan] (showing {len(sample_migrated)}/{migrated_count} records that will be migrated):"
            )
            for m in sample_migrated:
                console.print(f"  [dim magenta]📄 {m['file_id'][:8]}...[/dim magenta]")
                console.print(f"    [red]Before:[/red] [dim]{m['old_path']}[/dim]")
                console.print(f"    [green]After: [/green] {m['new_path']}")
            if not verbose and migrated_count > 10:
                console.print(f"  [dim]... and {migrated_count - 10} more[/dim]")

        if confirm:
            console.print(
                f"\n[bold green]✓ Successfully migrated {migrated_count} records[/bold green]"
            )
        else:
            # Always show dry run warning prominently
            console.print(
                "\n[bold cyan]ℹ️  DRY RUN MODE - No changes made to database[/bold cyan]"
            )
            console.print("[dim]Use --confirm to actually migrate these records.[/dim]")
            if migrated_count == 0:
                console.print("\n[yellow]No records to migrate.[/yellow]")
    # end of session's lifecycle


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to run"
    )

    # check 子命令
    subparsers.add_parser("check", help="Check migration status (no parameters needed)")

    # migrate 子命令
    migrate_parser = subparsers.add_parser(
        "migrate", help="Migrate absolute paths to relative paths"
    )
    migrate_parser.add_argument(
        "--uploads-dir",
        "-d",
        action="append",
        dest="uploads_dirs",
        help="Uploads directory to convert to relative paths. "
        "Required - specify the path(s) where files were stored. "
        "Can be specified multiple times (e.g., -d /old1 -d /old2).",
        metavar="PATH",
        required=True,
    )
    migrate_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually modify database (default is dry-run mode)",
    )
    migrate_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all records instead of just first 10",
    )
    migrate_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1000,
        metavar="N",
        help="Batch size for streaming and commits (default: 1000)",
    )

    return parser.parse_args()


def main():
    args = setup_args()
    try:
        if args.command == "check":
            check()
        elif args.command == "migrate":
            migrate(
                uploads_dirs=args.uploads_dirs,
                batch_size=args.batch_size,
                confirm=args.confirm,
                verbose=args.verbose,
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        console.print(f"\n[bold red]❌ Failed: {e}[/bold red]")
        sys.exit(2)


if __name__ == "__main__":
    main()
