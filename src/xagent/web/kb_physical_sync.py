"""KB collection physical directory sync: file lock + trash (rename-delete).

Provides:
- File system lock around collection directory operations to avoid concurrent
  delete/rename conflicts.
- Rename-to-trash instead of rmtree for delete: short inconsistent window
  and recoverable before cleanup via external scheduler (cron).
"""

import logging
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from filelock import FileLock, Timeout

from ..core.workspace import scoped_user_root

logger = logging.getLogger(__name__)

# Default lock timeout (seconds) when acquiring collection dir lock
DEFAULT_LOCK_TIMEOUT = 15.0

# Serialize long-running KB operations (delete vs ingest) on the same collection
# directory. Short timeouts cause false 409s when another request holds the lock
# for minutes (e.g. ingestion). Override via XAGENT_KB_COLLECTION_LOCK_TIMEOUT_SEC.
# See https://github.com/xorbitsai/xagent/issues/135 (control-plane / vector split).
DEFAULT_KB_COLLECTION_LOCK_TIMEOUT = float(
    os.environ.get("XAGENT_KB_COLLECTION_LOCK_TIMEOUT_SEC", "3600")
)

# Trash directory name under uploads (same volume for atomic rename)
TRASH_SUBDIR = ".trash"


def _lock_file_path_for_collection_dir(collection_dir: Path) -> Path:
    """Path to the lock file for a collection directory.

    Lock file is placed in the parent of the collection dir to avoid
    creating anything inside the collection. Format: .lock_<collection_name>
    """
    parent = collection_dir.parent
    name = collection_dir.name
    if not name:
        name = "root"
    return parent / f".lock_{name}"


@contextmanager
def collection_physical_lock(
    collection_dir: Path,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> Generator[None, None, None]:
    """Acquire an exclusive file system lock for operating on a collection directory.

    Use this around any physical operation (delete/rename) on the collection
    directory to prevent concurrent modifications (e.g. double delete, rename vs delete).

    Args:
        collection_dir: The collection directory path (e.g. uploads/user_1/my_coll).
        timeout: Max seconds to wait for the lock.

    Yields:
        None; hold the lock for the duration of the context.

    Raises:
        Timeout: If the lock cannot be acquired within timeout (caller may
            map to HTTP 409 or 503).
    """
    lock_path = _lock_file_path_for_collection_dir(collection_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # FileLock creates a .lock file alongside the given path
    lock = FileLock(str(lock_path) + ".lock")
    try:
        lock.acquire(timeout=timeout)
        try:
            yield
        finally:
            lock.release()
    except Timeout:
        logger.warning(
            "Collection directory lock timeout: %s (timeout=%.1fs)",
            collection_dir,
            timeout,
        )
        raise


def get_trash_path(
    uploads_dir: Path,
    user_id: int,
    collection_name: str,
) -> Path:
    """Return a unique path under uploads/.trash/user_{id}/ for this collection.

    Same volume as uploads so that rename is atomic. Name includes timestamp
    and short uuid to avoid collisions.
    """
    trash_base = scoped_user_root(uploads_dir / TRASH_SUBDIR, user_id)
    trash_base.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8]
    return trash_base / f"{collection_name}_{ts}_{uid}"


def move_collection_dir_to_trash(
    collection_dir: Path,
    uploads_dir: Path,
    user_id: int,
    collection_name: str,
) -> Path:
    """Move collection directory to trash. Caller must hold lock.

    Prefer atomic rename when possible; fall back to cross-device move when needed.

    Returns:
        The trash path to which the directory was renamed.
    """
    import shutil

    trash_path = get_trash_path(uploads_dir, user_id, collection_name)
    shutil.move(str(collection_dir), str(trash_path))
    logger.info(
        "Collection directory moved to trash: %s -> %s",
        collection_dir,
        trash_path,
    )
    return trash_path


def cleanup_trash(
    uploads_dir: Path,
    older_than_seconds: float = 7 * 24 * 3600,
) -> int:
    """Remove trash directories older than the given age. Safe to run as cron/scheduler.

    Args:
        uploads_dir: Base uploads path (contains .trash).
        older_than_seconds: Delete only dirs whose mtime is older than this (default 7 days).

    Returns:
        Number of trash directories removed.
    """
    trash_root = uploads_dir / TRASH_SUBDIR
    if not trash_root.exists():
        return 0
    import shutil

    now = time.time()
    removed = 0
    for user_dir in trash_root.iterdir():
        if not user_dir.is_dir():
            continue
        for item in user_dir.iterdir():
            if not item.is_dir():
                continue
            try:
                if now - item.stat().st_mtime >= older_than_seconds:
                    shutil.rmtree(item)
                    removed += 1
                    logger.info("Trash cleaned: %s", item)
            except OSError as e:
                logger.warning("Failed to remove trash dir %s: %s", item, e)
    return removed
