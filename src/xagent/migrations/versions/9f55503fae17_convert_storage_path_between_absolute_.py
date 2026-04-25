"""convert storage_path between absolute and relative

Alembic migration to convert storage_path between absolute and relative formats.
- Upgrade: absolute -> relative (using current UPLOADS_DIR)
- Downgrade: relative -> absolute (using current UPLOADS_DIR)

Current state (before upgrade): storage_path stores absolute paths like "/uploads/user_1/web_task_123/output/file.txt"
After upgrade: storage_path stores relative paths like "web_task_123/output/file.txt"

If automatic conversion fails (e.g., uploads_dir moved), users can run:
  python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path --confirm

Revision ID: 9f55503fae17
Revises: 20260403_add_user_tool_configs
Create Date: 2026-04-09 17:04:20.727649

"""

from pathlib import PurePosixPath, PureWindowsPath
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from dotenv import load_dotenv
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.
revision: str = "9f55503fae17"
down_revision: Union[str, None] = "20260403_add_user_tool_configs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Batch size for streaming and commits (same as manual script)
BATCH_SIZE = 1000


def upgrade() -> None:
    """Convert absolute paths to relative paths.

    Current state: storage_path = "/uploads/user_1/web_task_123/output/file.txt"
    After upgrade: storage_path = "web_task_123/output/file.txt"
    """

    load_dotenv()

    from xagent.config import get_uploads_dir
    from xagent.web.models.uploaded_file import UploadedFile

    # Get session directly from alembic connection
    with Session(bind=op.get_bind()) as session:
        # Use yield_per for streaming to avoid loading all records into memory
        stmt = sa.select(UploadedFile).execution_options(yield_per=BATCH_SIZE)

        count = 0
        for record in session.execute(stmt).scalars():
            storage_str = str(record.storage_path)

            # Detect path type (Windows vs Unix)
            is_win_style = len(storage_str) >= 2 and storage_str[1] == ":"

            if is_win_style:
                path_obj = PureWindowsPath(storage_str)
            else:
                path_obj = PurePosixPath(storage_str)

            # Skip if already relative
            if not path_obj.is_absolute():
                continue

            # Convert absolute to relative
            try:
                user_root = get_uploads_dir() / f"user_{record.user_id}"
                relative_path = path_obj.relative_to(user_root)
                record.storage_path = relative_path.as_posix()  # pyright: ignore[reportAttributeAccessIssue]
            except ValueError as e:
                # Path outside uploads_dir - keep as absolute but log the issue
                print(
                    f"Warning: Could not convert {storage_str} to relative path (outside UPLOADS_DIR): {e}"
                )
                print()
                print("Some paths could not be converted automatically.")
                print(
                    "This usually happens when XAGENT_UPLOADS_DIR has been changed multiple times."
                )
                print("Please use the manual migration tool to fix these paths:")
                print("  python scripts/migrate_uploads_file_abs_path.py check")
                print(
                    "  python scripts/migrate_uploads_file_abs_path.py migrate -d /old/uploads/path --confirm"
                )
                print()
                print(
                    "See scripts/migrate_uploads_file_abs_path.README.md for details."
                )

            count += 1
            # Batch commit with progress
            if count % BATCH_SIZE == 0:
                session.commit()
                print(f"Upgrade: Processed {count} records...")

        # Final commit for remaining records
        session.commit()
        if count > 0:
            print(f"Upgrade complete: {count} records converted to relative paths")


def downgrade() -> None:
    """Convert relative paths back to absolute paths.

    After upgrade state: storage_path = "web_task_123/output/file.txt"
    After downgrade: storage_path = "/uploads/user_1/web_task_123/output/file.txt"
    """

    load_dotenv()

    from xagent.config import get_uploads_dir
    from xagent.web.models.uploaded_file import UploadedFile

    # Get session directly from alembic connection
    with Session(bind=op.get_bind()) as session:
        # Use yield_per for streaming to avoid loading all records into memory
        stmt = sa.select(UploadedFile).execution_options(yield_per=BATCH_SIZE)

        count = 0
        for record in session.execute(stmt).scalars():
            storage_str = str(record.storage_path)

            # Detect path type (Windows vs Unix)
            is_win_style = len(storage_str) >= 2 and storage_str[1] == ":"

            if is_win_style:
                path_obj = PureWindowsPath(storage_str)
            else:
                path_obj = PurePosixPath(storage_str)

            # Skip if already absolute
            if path_obj.is_absolute():
                continue

            # Convert relative to absolute
            user_root = get_uploads_dir() / f"user_{record.user_id}"
            absolute_path = user_root / path_obj
            record.storage_path = str(absolute_path)  # pyright: ignore[reportAttributeAccessIssue]

            count += 1
            # Batch commit with progress
            if count % BATCH_SIZE == 0:
                session.commit()
                print(f"Downgrade: Processed {count} records...")

        # Final commit for remaining records
        session.commit()
        if count > 0:
            print(f"Downgrade complete: {count} records converted to absolute paths")
