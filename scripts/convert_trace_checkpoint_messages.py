#!/usr/bin/env python3
"""Convert historical checkpoint payloads to refs.

This script rewrites legacy checkpoint trace rows whose optimized storage
fields are still inline. New refs-format rows are left unchanged.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_path in (SRC_ROOT, PROJECT_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger("convert_trace_checkpoint_messages")


@dataclass
class ConversionStats:
    scanned_rows: int = 0
    converted_rows: int = 0
    already_refs_rows: int = 0
    skipped_rows: int = 0
    error_rows: int = 0
    inline_payload_bytes: int = 0
    refs_payload_bytes: int = 0

    @property
    def estimated_payload_bytes_saved(self) -> int:
        return max(0, self.inline_payload_bytes - self.refs_payload_bytes)


def convert_trace_checkpoint_messages(
    db: "Session",
    *,
    dry_run: bool = True,
    batch_size: int = 500,
    task_id: int | None = None,
    limit: int | None = None,
    start_id: int = 0,
) -> ConversionStats:
    """Convert legacy inline checkpoint payloads in trace_events.

    ``limit`` bounds the number of system trace rows scanned, not the number of
    rows converted.
    """

    from xagent.web.models.task import TraceEvent
    from xagent.web.services.trace_message_storage import (
        checkpoint_storage_payload_bytes,
        encode_checkpoint_data_for_storage,
        get_checkpoint_storage_state,
    )

    stats = ConversionStats()
    last_seen_id = start_id
    remaining = limit

    while remaining is None or remaining > 0:
        page_size = batch_size if remaining is None else min(batch_size, remaining)
        filters = [
            TraceEvent.id > last_seen_id,
            TraceEvent.event_type == "system_update_general",
        ]
        if task_id is not None:
            filters.append(TraceEvent.task_id == task_id)

        rows = (
            db.query(TraceEvent)
            .filter(*filters)
            .order_by(TraceEvent.id.asc())
            .limit(page_size)
            .all()
        )
        if not rows:
            break

        converted_in_batch = 0
        for row in rows:
            stats.scanned_rows += 1
            last_seen_id = int(row.id)
            data = row.data if isinstance(row.data, dict) else None
            state = get_checkpoint_storage_state(data)
            if state == "refs":
                stats.already_refs_rows += 1
                continue
            if state == "none":
                stats.skipped_rows += 1
                continue

            try:
                assert isinstance(data, dict)
                with db.begin_nested():
                    inline_payload_bytes = checkpoint_storage_payload_bytes(data)
                    encoded = encode_checkpoint_data_for_storage(
                        db,
                        task_id=int(row.task_id),
                        data=data,
                    )
                    encoded_state = get_checkpoint_storage_state(encoded)
                    if encoded_state != "refs":
                        stats.skipped_rows += 1
                        continue

                    if not dry_run:
                        row.data = encoded
                stats.converted_rows += 1
                converted_in_batch += 1
                stats.inline_payload_bytes += inline_payload_bytes
                stats.refs_payload_bytes += checkpoint_storage_payload_bytes(encoded)
            except Exception:
                stats.error_rows += 1
                logger.exception(
                    "Failed to convert trace event id=%s task_id=%s",
                    row.id,
                    row.task_id,
                )
                continue

        if dry_run:
            db.rollback()
        elif converted_in_batch:
            db.commit()
        else:
            db.rollback()

        if remaining is not None:
            remaining -= len(rows)

        logger.info(
            "Scanned %s rows, converted %s, already refs %s, skipped %s",
            stats.scanned_rows,
            stats.converted_rows,
            stats.already_refs_rows,
            stats.skipped_rows,
        )

    return stats


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded environment from %s", env_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert legacy checkpoint payloads to trace storage refs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run against the configured database
  python scripts/convert_trace_checkpoint_messages.py

  # Convert all historical checkpoints
  python scripts/convert_trace_checkpoint_messages.py --execute

  # Convert one task in small batches
  python scripts/convert_trace_checkpoint_messages.py --execute --task-id 467 --batch-size 100
        """,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Write converted trace rows. Default is dry-run.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without writing changes. This is the default.",
    )
    parser.add_argument(
        "--database-url",
        help="Override DATABASE_URL for this run.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        help="Only convert checkpoints for one task.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of system trace rows to scan per batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of system trace rows to scan.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Only scan trace_events rows with id greater than this value.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    if args.batch_size < 1:
        logger.error("--batch-size must be positive")
        return 2
    if args.limit is not None and args.limit < 1:
        logger.error("--limit must be positive")
        return 2
    if args.start_id < 0:
        logger.error("--start-id cannot be negative")
        return 2

    dry_run = not args.execute
    _load_dotenv()
    from xagent.web.models.database import get_session_local, init_db

    init_db(args.database_url)
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        if dry_run:
            logger.info("Dry-run mode: no database changes will be committed")
        else:
            logger.info("Execute mode: converted rows will be committed")

        stats = convert_trace_checkpoint_messages(
            db,
            dry_run=dry_run,
            batch_size=args.batch_size,
            task_id=args.task_id,
            limit=args.limit,
            start_id=args.start_id,
        )
    finally:
        db.close()

    logger.info("=" * 72)
    logger.info("Trace checkpoint message conversion summary")
    logger.info("Rows scanned:         %s", stats.scanned_rows)
    logger.info("Rows converted:       %s", stats.converted_rows)
    logger.info("Rows already refs:    %s", stats.already_refs_rows)
    logger.info("Rows skipped:         %s", stats.skipped_rows)
    logger.info("Rows errored:         %s", stats.error_rows)
    logger.info("Inline payload bytes: %s", stats.inline_payload_bytes)
    logger.info("Refs payload bytes:   %s", stats.refs_payload_bytes)
    logger.info("Estimated trace row saved: %s", stats.estimated_payload_bytes_saved)
    logger.info("=" * 72)
    return 1 if stats.error_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
