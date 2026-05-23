"""Cross-process locks for document generation (chunk + embedding) scopes.

Serialises work keyed by ``(collection, doc_id, parse_hash, user scope)`` so
concurrent re-chunk / re-embed runs for the same logical document cannot race
across chunk replacement and embedding writes (PR #202).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple

from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

# Re-entrant per-process locks; paired with FileLock for multi-worker safety.
_thread_locks: dict[Tuple[str, str, str, str, str], threading.RLock] = {}
_thread_locks_guard = threading.Lock()

_lock_depth = threading.local()

DEFAULT_GENERATION_LOCK_TIMEOUT = float(
    os.environ.get("XAGENT_INGESTION_GENERATION_LOCK_TIMEOUT_SEC", "3600")
)


def generation_lock_key(
    collection: str,
    doc_id: str,
    parse_hash: str,
    user_id: Optional[int],
    is_admin: bool,
) -> Tuple[str, str, str, str, str]:
    """Build the canonical lock key for a document generation scope."""
    return (
        collection,
        doc_id,
        parse_hash,
        str(user_id) if user_id is not None else "",
        "admin" if is_admin else "user",
    )


def _get_thread_lock(key: Tuple[str, str, str, str, str]) -> threading.RLock:
    if key in _thread_locks:
        return _thread_locks[key]
    with _thread_locks_guard:
        if key not in _thread_locks:
            _thread_locks[key] = threading.RLock()
        return _thread_locks[key]


def _locks_directory() -> Path:
    lancedb_dir = os.getenv("LANCEDB_DIR")
    if not lancedb_dir:
        from xagent.providers.vector_store.lancedb import LanceDBConnectionManager

        lancedb_dir = LanceDBConnectionManager.get_default_lancedb_dir()
    locks_dir = Path(lancedb_dir) / ".ingestion_generation_locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    return locks_dir


def _file_lock_path(key: Tuple[str, str, str, str, str]) -> Path:
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
    return _locks_directory() / f"{digest}.lock"


@contextmanager
def generation_ingestion_lock(
    collection: str,
    doc_id: str,
    parse_hash: str,
    user_id: Optional[int],
    is_admin: bool,
    *,
    timeout: float = DEFAULT_GENERATION_LOCK_TIMEOUT,
) -> Iterator[None]:
    """Acquire a generation-scope lock (in-process re-entrant + cross-process).

    Nested calls for the same scope from one thread only acquire the file lock
    once, so ``process_document`` can call ``replace_chunks`` safely.
    """
    key = generation_lock_key(collection, doc_id, parse_hash, user_id, is_admin)
    thread_lock = _get_thread_lock(key)

    depth = getattr(_lock_depth, "value", 0)
    acquire_file = depth == 0
    _lock_depth.value = depth + 1

    thread_lock.acquire()
    file_lock: Optional[FileLock] = None
    try:
        if acquire_file:
            file_lock = FileLock(str(_file_lock_path(key)))
            try:
                file_lock.acquire(timeout=timeout)
            except Timeout as exc:
                logger.warning(
                    "Generation lock timeout: collection=%s doc_id=%s parse_hash=%s "
                    "timeout=%.1fs",
                    collection,
                    doc_id,
                    parse_hash,
                    timeout,
                )
                raise Timeout(
                    f"Timed out waiting for ingestion generation lock on "
                    f"{collection}/{doc_id}/{parse_hash}"
                ) from exc
        yield
    finally:
        if acquire_file and file_lock is not None:
            file_lock.release()
        _lock_depth.value = max(getattr(_lock_depth, "value", 1) - 1, 0)
        thread_lock.release()
