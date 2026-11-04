#!/usr/bin/env python3
"""Fix user_id field type in LanceDB tables.

The previous migration created user_id fields with type 'null' instead of 'int64'.
This script fixes the type by using the correct schema from schema_manager.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from ...config import get_lancedb_path as get_config_lancedb_path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import pyarrow.parquet as pq
from lancedb import connect

logger = logging.getLogger(__name__)


def fix_table_type(
    conn,
    table_name: str,
    backup_dir: Path,
    execute: bool = False,
) -> bool:
    """Fix user_id field type for a table.

    Args:
        conn: LanceDB connection
        table_name: Name of table to fix
        backup_dir: Directory for backup files
        execute: If False, only simulate

    Returns:
        True if successful
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing table: {table_name}")
    logger.info(f"{'=' * 60}")

    # Check if table exists
    try:
        table = conn.open_table(table_name)
    except Exception as e:
        logger.error(f"✗ Could not open table '{table_name}': {e}")
        return False

    # Step 1: Export existing data to Parquet
    logger.info(f"Step 1: Exporting data from '{table_name}'...")
    try:
        data = table.to_arrow()
        backup_file = backup_dir / f"{table_name}.parquet"

        if execute:
            pq.write_table(data, backup_file)
            logger.info(f"  ✓ Exported to {backup_file}")
            logger.info(f"    Rows: {len(data)}")
        else:
            logger.info(f"  [DRY RUN] Would export to {backup_file}")
            logger.info(f"    Would export {len(data)} rows")

    except Exception as e:
        logger.error(f"✗ Export failed: {e}")
        return False

    # Step 2: Drop old table
    logger.info(f"Step 2: Dropping old table '{table_name}'...")
    if execute:
        try:
            conn.drop_table(table_name)
            logger.info(f"  ✓ Dropped table '{table_name}'")
        except Exception as e:
            logger.error(f"✗ Drop failed: {e}")
            return False
    else:
        logger.info(f"  [DRY RUN] Would drop table '{table_name}'")

    # Step 3: Import data with correct schema
    logger.info("Step 3: Importing data with corrected user_id type...")
    if execute:
        try:
            import pandas as pd

            # Read backup data
            df = pd.read_parquet(backup_file)

            # Ensure user_id column exists with proper Int64 dtype
            if "user_id" not in df.columns:
                # Add user_id column with None values
                df["user_id"] = pd.array([None] * len(df), dtype="Int64")
            else:
                # Convert existing user_id column to Int64
                df["user_id"] = df["user_id"].astype("Int64")

            # Convert to pyarrow table with proper schema
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
            from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
                ensure_chunks_table,
                ensure_documents_table,
                ensure_embeddings_table,
                ensure_ingestion_runs_table,
                ensure_parses_table,
            )

            # Create the table with proper schema first
            if table_name == "documents":
                ensure_documents_table(conn)
            elif table_name == "parses":
                ensure_parses_table(conn)
            elif table_name == "chunks":
                ensure_chunks_table(conn)
            elif table_name == "ingestion_runs":
                ensure_ingestion_runs_table(conn)
            elif table_name.startswith("embeddings_"):
                model_tag = table_name.replace("embeddings_", "")
                ensure_embeddings_table(conn, model_tag)
            else:
                # For other tables, just create with the dataframe
                conn.create_table(table_name, data=df)
                logger.info(f"  ✓ Imported {len(df)} rows with Int64 user_id")
                return True

            # Now add the data using add() which respects the existing schema
            table = conn.open_table(table_name)
            table.add(df)

            logger.info(f"  ✓ Imported {len(df)} rows with Int64 user_id")

        except Exception as e:
            logger.error(f"✗ Import failed: {e}")
            logger.error(f"  Backup available at: {backup_file}")
            import traceback

            traceback.print_exc()
            return False
    else:
        logger.info("  [DRY RUN] Would import data with Int64 user_id")

    logger.info(f"✓ Migration complete for table '{table_name}'")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix user_id field type in LanceDB tables"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the fix (default: dry-run)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to LanceDB database",
    )
    parser.add_argument(
        "--cleanup-backup",
        action="store_true",
        help="Automatically remove backup directory after successful migration",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    db_path = args.db_path or str(get_config_lancedb_path())
    logger.info(f"LanceDB path: {db_path}")

    if not args.execute:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("Use --execute to actually perform the fix")
        logger.info("=" * 60)

    try:
        conn = connect(db_path)
        logger.info("✓ Connected to database")
    except Exception as e:
        logger.error(f"✗ Failed to connect: {e}")
        sys.exit(1)

    tables_to_fix = [
        "documents",
        "parses",
        "chunks",
        "ingestion_runs",
        "embeddings_text_embedding_v4",
    ]

    # Create backup directory
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"./lancedb_fix_backup_{timestamp}")
    if args.execute:
        backup_dir.mkdir(exist_ok=True)
        logger.info(f"\nBackup directory: {backup_dir}")

    success_count = 0
    failed_tables = []

    for table_name in tables_to_fix:
        if fix_table_type(conn, table_name, backup_dir, args.execute):
            success_count += 1
        else:
            failed_tables.append(table_name)

    logger.info("\n" + "=" * 60)
    logger.info("Fix Summary")
    logger.info("=" * 60)
    logger.info(f"Total tables: {len(tables_to_fix)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(failed_tables)}")

    if failed_tables:
        logger.error(f"Failed tables: {failed_tables}")
        sys.exit(1)

    # Cleanup backup if requested and migration was successful
    if args.execute and args.cleanup_backup and backup_dir.exists():
        logger.info(f"\n[*] Cleaning up backup directory: {backup_dir}")
        shutil.rmtree(backup_dir)
        logger.info("✓ Backup directory removed")
    elif args.execute and backup_dir.exists():
        logger.info(f"\n[*] Backup directory preserved at: {backup_dir.resolve()}")
        logger.info("[*] You can remove it manually if migration is verified:")


if __name__ == "__main__":
    main()
