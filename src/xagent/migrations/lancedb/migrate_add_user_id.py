#!/usr/bin/env python3
"""LanceDB schema migration: Add user_id field to existing tables.

This script migrates existing LanceDB tables to include the user_id field.
Existing data will have user_id set to NULL (accessible to all users).

Usage:
    python -m xagent.migrations.lancedb.migrate_add_user_id --dry-run
    python -m xagent.migrations.lancedb.migrate_add_user_id --execute
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import pyarrow.parquet as pq
from lancedb import connect
from lancedb.db import DBConnection

from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import (
    list_table_names,
)

logger = logging.getLogger(__name__)


def get_lancedb_path() -> str:
    """Get LanceDB database path from configuration."""
    from dotenv import load_dotenv

    from xagent.config import get_lancedb_path as get_config_lancedb_path

    load_dotenv()

    # Use centralized config function
    return str(get_config_lancedb_path())


def get_embeddings_tables(conn: DBConnection) -> list[str]:
    """Get list of embeddings tables (pattern: embeddings_*)"""
    try:
        existing_tables = list_table_names(conn)
        embeddings_tables = [t for t in existing_tables if t.startswith("embeddings_")]
        return embeddings_tables
    except Exception as e:
        logger.warning(f"Could not list tables: {e}")
        return []


def migrate_table(
    conn: DBConnection,
    table_name: str,
    backup_dir: Path,
    execute: bool = False,
) -> bool:
    """Migrate a single table to add user_id field.

    Args:
        conn: LanceDB connection
        table_name: Name of table to migrate
        backup_dir: Directory for backup files
        execute: If False, only simulate the migration

    Returns:
        True if migration succeeded or was skipped
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing table: {table_name}")
    logger.info(f"{'=' * 60}")

    # Check if table exists
    try:
        table = conn.open_table(table_name)
        existing_schema = table.schema
        field_names = {field.name for field in existing_schema}

        # Check if user_id already exists
        if "user_id" in field_names:
            logger.info(f"✓ Table '{table_name}' already has user_id field - skipping")
            return True

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

    # Step 3: Import data with user_id=NULL
    logger.info("Step 3: Importing data with user_id field...")
    if execute:
        try:
            import pandas as pd

            # Read backup data
            df = pd.read_parquet(backup_file)

            # Add user_id column with NULL values
            df["user_id"] = None

            # Create new table with updated schema
            # Note: This will use the schema_manager to create the proper schema
            conn.create_table(table_name, data=df)

            logger.info(f"  ✓ Imported {len(df)} rows with user_id=NULL")

        except Exception as e:
            logger.error(f"✗ Import failed: {e}")
            logger.error(f"  Backup available at: {backup_file}")
            return False
    else:
        logger.info("  [DRY RUN] Would import data with user_id=NULL")

    logger.info(f"✓ Migration complete for table '{table_name}'")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate LanceDB tables to add user_id field"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration (default: dry-run)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to LanceDB database (default: from LANCEDB_PATH env or ./data/lancedb)",
    )
    parser.add_argument(
        "--tables",
        type=str,
        nargs="+",
        default=None,
        help="Specific tables to migrate (default: all known tables)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding tables (will be detected automatically)",
    )
    parser.add_argument(
        "--cleanup-backup",
        action="store_true",
        help="Automatically remove backup directory after successful migration",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # Get database path
    db_path = args.db_path or get_lancedb_path()
    logger.info(f"LanceDB path: {db_path}")

    if not args.execute:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("Use --execute to actually perform the migration")
        logger.info("=" * 60)

    # Connect to database
    try:
        conn = connect(db_path)
        logger.info("✓ Connected to database")
    except Exception as e:
        logger.error(f"✗ Failed to connect to database: {e}")
        sys.exit(1)

    # List existing tables
    try:
        existing_tables = list_table_names(conn)
        logger.info(f"\nExisting tables: {existing_tables}")
    except Exception as e:
        logger.error(f"✗ Could not list tables: {e}")
        sys.exit(1)

    # Determine tables to migrate
    core_tables = [
        "documents",
        "parses",
        "chunks",
        "main_pointers",
        "prompt_templates",
        "ingestion_runs",
    ]

    embeddings_tables = get_embeddings_tables(conn) if not args.skip_embeddings else []

    if args.tables:
        # User specified tables
        tables_to_migrate = [t for t in args.tables if t in existing_tables]
    else:
        # All tables
        tables_to_migrate = [t for t in core_tables if t in existing_tables]
        tables_to_migrate.extend(embeddings_tables)

    if not tables_to_migrate:
        logger.info("No tables to migrate.")
        sys.exit(0)

    logger.info(f"\nTables to migrate: {tables_to_migrate}")

    # Create backup directory
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"./lancedb_backup_{timestamp}")
    if args.execute:
        backup_dir.mkdir(exist_ok=True)
        logger.info(f"\nBackup directory: {backup_dir}")

    # Migrate each table
    success_count = 0
    failed_tables = []

    for table_name in tables_to_migrate:
        if migrate_table(conn, table_name, backup_dir, execute=args.execute):
            success_count += 1
        else:
            failed_tables.append(table_name)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info(f"Total tables: {len(tables_to_migrate)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(failed_tables)}")

    if failed_tables:
        logger.error(f"Failed tables: {failed_tables}")
        sys.exit(1)
    elif not args.execute:
        logger.info("\n[DRY RUN] Complete. Run with --execute to perform migration.")
    else:
        logger.info("\n✓ Migration complete!")
        logger.info(f"Backups saved to: {backup_dir}")

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
