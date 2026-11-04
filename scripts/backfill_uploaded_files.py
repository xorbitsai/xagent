#!/usr/bin/env python3
"""
Backfill uploaded files to database.

This script scans the uploads directory and creates database records
for any files that are not already registered in the uploaded_files table.

Run this script once during upgrade to backfill legacy files.

Usage:
    python scripts/backfill_uploaded_files.py [--dry-run] [--user-id USER_ID]

Options:
    --dry-run        Scan and report without making changes
    --user-id USER_ID  Only backfill files for specific user (default: all users)
    --once            Only run if backfill hasn't been completed before
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f".env file not found at {env_path}")
except ImportError:
    logger.warning("python-dotenv not available, skipping .env loading")


def get_database_url():
    """Get database URL from environment or default."""
    # Try common environment variables
    db_path = os.environ.get("DATABASE_URL")
    if db_path:
        # If it looks like a URL (contains ://), return it directly
        if "://" in db_path:
            return db_path

        # If it's a file path, check if it exists and wrap with sqlite:///
        if Path(db_path).exists():
            return f"sqlite:///{db_path}"

    # Try default location
    default_db = Path(__file__).parent.parent / "xagent.db"
    if default_db.exists():
        return f"sqlite:///{default_db}"

    # Try in data directory
    data_db = Path(__file__).parent.parent / "data" / "xagent.db"
    if data_db.exists():
        return f"sqlite:///{data_db}"

    raise FileNotFoundError(
        "Database file not found. Please set DATABASE_URL environment variable."
    )


def scan_user_directory(user_root: Path, db_session) -> dict:
    """Scan a user's directory for unregistered files."""
    from xagent.web.models.task import Task
    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.models.user import User

    # Check if user exists
    try:
        user_id = int(user_root.name.replace("user_", "", 1))
    except ValueError:
        return {"error": f"Invalid user directory: {user_root.name}", "created": 0}

    user = db_session.query(User).filter(User.id == user_id).first()
    if not user:
        return {"error": f"User {user_id} not found in database", "created": 0}

    # Get existing file paths in database
    existing_paths = {
        row[0]
        for row in db_session.query(UploadedFile.storage_path)
        .filter(UploadedFile.user_id == user_id)
        .all()
    }

    # Scan for files
    created = 0
    skipped = 0
    errors = []

    if not user_root.exists():
        return {"error": f"User directory not found: {user_root}", "created": 0}

    for file_path in user_root.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip hidden files and cache directories
        if any(part.startswith(".") for part in file_path.parts):
            continue
        if "__pycache__" in file_path.parts or "node_modules" in file_path.parts:
            continue

        storage_path = str(file_path)
        if storage_path in existing_paths:
            skipped += 1
            continue

        # Infer task_id from path
        task_id = None
        try:
            rel_parts = file_path.relative_to(user_root).parts
            if rel_parts:
                first_part = rel_parts[0]
                if first_part.startswith("web_task_"):
                    task_id_part = first_part.replace("web_task_", "", 1)
                    task_id = int(task_id_part)
                    # Verify task exists and belongs to user
                    task = (
                        db_session.query(Task)
                        .filter(Task.id == task_id, Task.user_id == user_id)
                        .first()
                    )
                    if not task:
                        task_id = None
        except (ValueError, IndexError):
            pass

        # Create file record
        try:
            import mimetypes

            mime_type, _ = mimetypes.guess_type(file_path.name)
            if not mime_type:
                mime_type = "application/octet-stream"

            file_record = UploadedFile(
                user_id=user_id,
                task_id=task_id,
                filename=file_path.name,
                storage_path=storage_path,
                mime_type=mime_type,
                file_size=file_path.stat().st_size,
            )
            db_session.add(file_record)
            created += 1
            logger.info(f"Registered: {storage_path}")
        except Exception as e:
            errors.append(f"{storage_path}: {str(e)}")

    return {
        "user_id": user_id,
        "created": created,
        "skipped": skipped,
        "errors": errors,
    }


def backfill_all_users(
    dry_run: bool = False, user_id: int | None = None, db_session=None
):
    """Backfill files for all users or a specific user."""
    from xagent.config import get_uploads_dir

    uploads_dir = get_uploads_dir()
    if not uploads_dir.exists():
        logger.error(f"Uploads directory not found: {uploads_dir}")
        return

    # Collect target user directories
    user_dirs = []
    if user_id:
        user_dir = uploads_dir / f"user_{user_id}"
        if user_dir.exists():
            user_dirs.append(user_dir)
        else:
            logger.error(f"User directory not found: {user_dir}")
            return
    else:
        user_dirs = sorted(uploads_dir.glob("user_*"))

    if not user_dirs:
        logger.warning("No user directories found")
        return

    logger.info(f"Found {len(user_dirs)} user directories to process")

    # Process each user directory
    total_created = 0
    total_skipped = 0
    all_errors = []

    for user_dir in user_dirs:
        logger.info(f"Processing {user_dir.name}...")
        result = scan_user_directory(user_dir, db_session)

        if "error" in result:
            logger.error(f"  {result['error']}")
            continue

        total_created += result["created"]
        total_skipped += result["skipped"]
        all_errors.extend(result["errors"])

        logger.info(f"  Created: {result['created']}, Skipped: {result['skipped']}")

        if result["errors"]:
            logger.warning(f"  Errors: {len(result['errors'])}")
            for error in result["errors"][:5]:  # Show first 5 errors
                logger.warning(f"    - {error}")

    # Commit changes
    if not dry_run and total_created > 0:
        try:
            db_session.commit()
            logger.info("✅ Database commit successful")
        except Exception as e:
            db_session.rollback()
            logger.error(f"❌ Database commit failed: {e}")
            raise
    elif dry_run and total_created > 0:
        logger.info("🔍 Dry run mode - no changes made")
    else:
        logger.info("ℹ️  No new files to register")

    # Summary
    logger.info("=" * 60)
    logger.info("Backfill Summary:")
    logger.info(f"  Users processed: {len(user_dirs)}")
    logger.info(f"  Files created:   {total_created}")
    logger.info(f"  Files skipped:  {total_skipped}")
    logger.info(f"  Errors:          {len(all_errors)}")
    logger.info("=" * 60)


def check_backfill_completion(db_session) -> bool:
    """Check if backfill has been completed before."""
    # This could check a flag in the database or a marker file
    # For now, we'll just check if there are any files in the database
    from xagent.web.models.uploaded_file import UploadedFile

    count = db_session.query(UploadedFile).count()
    return count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Backfill uploaded files to database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run to see what would be backfilled
    python scripts/backfill_uploaded_files.py --dry-run

    # Backfill all users
    python scripts/backfill_uploaded_files.py

    # Backfill specific user
    python scripts/backfill_uploaded_files.py --user-id 1

    # Run once (skip if already done)
    python scripts/backfill_uploaded_files.py --once
        """,
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Scan and report without making changes"
    )

    parser.add_argument(
        "--user-id", type=int, help="Only backfill files for specific user ID"
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Only run if backfill hasn't been completed before",
    )

    args = parser.parse_args()

    # Setup database connection
    try:
        db_url = get_database_url()
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(bind=engine)
        db_session = SessionLocal()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    try:
        # Check if should run (for --once flag)
        if args.once and check_backfill_completion(db_session):
            logger.info("ℹ️  Backfill already completed, skipping (--once flag set)")
            logger.info("   To force backfill, remove the --once flag")
            sys.exit(0)

        # Run backfill
        logger.info("Starting file backfill...")
        if args.dry_run:
            logger.info("🔍 DRY RUN MODE - No changes will be made")

        backfill_all_users(
            dry_run=args.dry_run, user_id=args.user_id, db_session=db_session
        )

        if not args.dry_run:
            logger.info("✅ Backfill completed successfully")

    except KeyboardInterrupt:
        logger.info("\n⚠️  Backfill interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Backfill failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db_session.close()


if __name__ == "__main__":
    main()
