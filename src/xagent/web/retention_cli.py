"""``xagent retention`` -- read-only retention diagnostics (#2562).

``preview`` answers the question the policy decision in #2567 is blocked on:
*if we set the retention period to N days, how much data does that actually
expire?* It reports, per candidate N, the eligible task count and the
``trace_events`` rows hanging off those tasks, using the same eligibility
predicate the purge (#2563) will use -- so the number an operator sees here
is the number the purge would act on, not an estimate computed a second way.

It deletes nothing and it locks nothing. The database is bound with
``configure_db(read_only=True)``, which makes read-only a database-enforced
property (a PostgreSQL ``READ ONLY`` transaction, a SQLite read-only URL)
rather than a promise this module keeps.

**Cost.** ``tasks.last_activity_at`` is deliberately unindexed in this
revision (see the column comment in ``models/task.py``), so each period costs
a sequential scan of ``tasks`` plus one over the matching ``trace_events``.
That is affordable for a diagnostic an operator runs by hand a few times, and
not affordable for a loop; the purge in #2563 adds the index along with the
scan that needs it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

from .models.database import configure_db, get_session_local
from .services.task_retention import (
    count_quiescent_tasks,
    count_retention_candidate_trace_events,
    count_retention_candidates,
    retention_cutoff,
)

#: Offered as the default sweep because they are the options on the table in
#: the #2567 decision matrix. Not a recommendation and not a configured value.
DEFAULT_PREVIEW_DAYS = (90, 180, 365)


def add_preview_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--days",
        type=int,
        action="append",
        dest="days",
        metavar="N",
        help=(
            "Retention period to evaluate, in days. Repeatable to compare "
            f"candidates. Default: {', '.join(str(d) for d in DEFAULT_PREVIEW_DAYS)}."
        ),
    )
    parser.add_argument(
        "--database-url",
        dest="database_url",
        help="Override the database to inspect. Defaults to the configured one.",
    )


def _format_table(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for index, row in enumerate(rows):
        lines.append(
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        )
        if index == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(row))))
    return "\n".join(lines)


def run_preview(args: argparse.Namespace) -> int:
    days_list = sorted(set(args.days or DEFAULT_PREVIEW_DAYS))
    invalid = [d for d in days_list if d < 0]
    if invalid:
        print(
            f"--days must not be negative; got {', '.join(str(d) for d in invalid)}.",
            file=sys.stderr,
        )
        return 2

    now = datetime.now(timezone.utc)
    # Ask the arithmetic itself rather than inventing a ceiling. Both legs can
    # overflow and they differ: ``timedelta(days=1_000_000_000)`` refuses to
    # construct, while ``days=999_999_999`` constructs and then overflows the
    # subtraction. Any hand-written bound would guess one of the two wrong.
    # The check stays here and not in ``retention_cutoff``: that is library
    # code, and #2563 wants the exception rather than a silent clamp.
    unrepresentable = []
    for days in days_list:
        try:
            retention_cutoff(now=now, days=days)
        except OverflowError:
            unrepresentable.append(days)
    if unrepresentable:
        print(
            "--days is too large to express as a date: "
            f"{', '.join(str(d) for d in unrepresentable)}.",
            file=sys.stderr,
        )
        return 2

    print(
        "Scanning an unindexed column, one sequential pass per period; "
        "prefer off-peak on a large deployment.",
        file=sys.stderr,
    )
    configure_db(args.database_url, read_only=True)
    sessions = get_session_local()
    with sessions() as db:
        quiescent = count_quiescent_tasks(db, now=now)
        rows: list[tuple[str, ...]] = [("days", "eligible tasks", "trace events")]
        for days in days_list:
            rows.append(
                (
                    str(days),
                    str(count_retention_candidates(db, now=now, days=days)),
                    str(count_retention_candidate_trace_events(db, now=now, days=days)),
                )
            )

    print(f"Evaluated at {now.isoformat()}")
    print(f"Quiescent tasks (terminal, no live lease, no pending command): {quiescent}")
    print()
    print(_format_table(rows))
    print()
    print(
        "Counts are what the purge would select, not what it has deleted: "
        "this command deletes nothing."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="xagent retention",
        description="Read-only conversation retention diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    xagent retention preview                          # compare 90/180/365 days
    xagent retention preview --days 30                # a single candidate
    xagent retention preview --days 90 --days 365     # compare two
        """,
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    preview = subcommands.add_parser(
        "preview",
        help="Report how many tasks and traces a retention period would expire.",
    )
    add_preview_arguments(preview)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.command == "preview":
        return run_preview(args)
    parser.error(f"unknown command: {args.command}")
    return 2
