"""Storage maintenance for the uploads tree.

Owns the orphaned temp-file sweep: which pathnames count as an abandoned
atomic-replace temp, how the uploads tree is traversed, and how the walk
unwinds on a shutdown signal. Lifecycle orchestration (scheduling the sweep,
signalling it, waiting for it) lives in ``web/app.py``.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


# WHY: a single very large flat directory must still be interruptible within
# the shutdown grace period, not only at directory boundaries -- checking
# every entry would add per-entry overhead for no benefit at typical sizes.
# The same chunk size bounds that directory's deletion phase below.
_TEMP_CLEANUP_STOP_CHECK_ENTRIES = 500


class _StopSignal(Protocol):
    """The only threading.Event behavior this walk actually needs.

    A Protocol (rather than threading.Event itself) so a lightweight test
    double satisfies the type without implementing set()/clear()/wait().
    """

    def is_set(self) -> bool: ...


def cleanup_orphaned_temp_files(
    upload_dir: Optional[Path] = None,
    *,
    stop_event: Optional[_StopSignal] = None,
) -> int:
    """Clean up orphaned temporary files from interrupted atomic replacements.

    Removes files matching patterns like:
    - *.tmp-replace (old pattern)
    - .*.tmp (new NamedTemporaryFile pattern)

    Deleting a temp whose writer stalled past the age threshold is possible,
    but bounded: every producer of these names is create-temp -> write ->
    atomic replace, so the worst case is that writer's replace failing, never
    a partial file at the target. A producer that writes in place voids this.

    Args:
        upload_dir: Base uploads directory to clean. If None, uses default uploads dir.
        stop_event: Optional cooperative stop flag, checked once per directory and
            periodically within a large directory's scan. When set, the walk
            unwinds early so a shutdown can interrupt a long sweep instead of
            the worker thread blocking process exit via asyncio.run()'s
            executor-thread join.

    Returns:
        Number of files cleaned up.
    """
    from ..config import get_uploads_dir

    base_dir = upload_dir or get_uploads_dir()
    if not base_dir.exists():
        return 0

    cleaned_count = 0
    now = time.time()

    def _is_orphaned_temp_name(filename: str) -> bool:
        # Old atomic-replace pattern (*.tmp-replace).
        if filename.endswith(".tmp-replace"):
            return True
        # New NamedTemporaryFile pattern: filename.XXXXXX.tmp (has multiple
        # extensions, so at least three dot-separated parts ending in ``tmp``).
        if filename.endswith(".tmp") and "." in filename[:-4]:
            parts = filename.split(".")
            if len(parts) >= 3 and parts[-1] == "tmp":
                return True
        return False

    # Walk the uploads tree with os.scandir, seeding the stack with a str path
    # and appending entry.path directly so no Path object is allocated per
    # entry on a large tree. Clean temp files older than 1 hour to avoid
    # deleting files that might still be in use; per-entry OSError is tolerated
    # below so a file vanishing mid-scan skips instead of aborting the sweep.
    root = str(base_dir)
    stack = [root]
    # WHY: track "did we stop early" as we go instead of re-reading stop_event
    # after the loop. The flag can be set by shutdown *while* the final
    # directory is being scanned, in which case the tree still gets fully
    # walked -- re-polling afterwards would report a completed sweep as
    # truncated and tell operators to expect leftover orphans that aren't there.
    stopped_early = False
    while stack:
        # Cooperative stop check; see the stop_event docstring above for why.
        if stop_event is not None and stop_event.is_set():
            stopped_early = True
            break
        current = stack.pop()
        # WHY: os.scandir() on this str path follows symlinks, and the DFS stack
        # can hold a queued pathname for the rest of the sweep -- so without a
        # fresh check, a directory swapped for a symlink after its parent's
        # is_dir(follow_symlinks=False) sends the sweep outside the uploads root.
        # os.walk() guards the same race the same way (CPython bpo-23605), at the
        # same cost of one lstat per directory. The root stays exempt because
        # os.walk() never tested `top` either, and a symlinked uploads root
        # (a mounted volume) is a legitimate deployment.
        if current != root and os.path.islink(current):
            logger.warning(
                "Not descending into %s: it was replaced by a symlink during the sweep",
                current,
            )
            continue
        try:
            scandir_it = os.scandir(current)
        except OSError as e:
            logger.warning("Failed to scan directory %s: %s", current, e)
            continue
        # WHY: matches are collected here and unlinked only after the directory
        # stream is closed below. Whether readdir() still returns the remaining
        # entries of a directory that was modified mid-iteration is unspecified
        # by POSIX, so unlinking inside the scan can silently skip siblings --
        # and this sweep runs once per process with no resumption, so a skipped
        # orphan is never reclaimed. Both stdlib walkers avoid this: os.walk
        # exhausts and closes the iterator before yielding, and shutil.rmtree
        # materializes list(scandir_it) before unlinking.
        victims: list[str] = []
        with scandir_it:
            entries_seen = 0
            while True:
                try:
                    entry = next(scandir_it)
                except StopIteration:
                    break
                except OSError as e:
                    # A directory-level failure mid-iteration (e.g. NFS ESTALE,
                    # or the directory removed mid-scan). Match os.walk's default
                    # tolerance: skip this directory instead of aborting the
                    # whole sweep.
                    logger.warning("Failed while scanning directory %s: %s", current, e)
                    break
                entries_seen += 1
                if (
                    stop_event is not None
                    and entries_seen % _TEMP_CLEANUP_STOP_CHECK_ENTRIES == 0
                    and stop_event.is_set()
                ):
                    stopped_early = True
                    break
                try:
                    is_directory = entry.is_dir(follow_symlinks=False)
                except OSError as e:
                    # WHY: split from the candidate probes below -- a failed
                    # directory probe silently drops every orphan beneath it,
                    # and this one-shot sweep has no resumption to reclaim them.
                    logger.warning(
                        "Skipping %s: directory probe failed, so any subtree "
                        "below it is not swept: %s",
                        entry.path,
                        e,
                    )
                    continue
                if is_directory:
                    stack.append(entry.path)
                    continue
                try:
                    # Name test first: it touches no filesystem, so the vast
                    # majority of entries are rejected before is_file()/stat().
                    if not _is_orphaned_temp_name(entry.name):
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if now - entry.stat().st_mtime <= 3600:  # 1 hour
                        continue
                except OSError as e:
                    # The entry vanished mid-scan (e.g. a concurrent replace);
                    # skip it rather than aborting the whole sweep.
                    logger.debug("Skipping temp-file candidate %s: %s", entry.path, e)
                    continue
                victims.append(entry.path)

        for victims_seen, victim in enumerate(victims):
            # WHY: an unpolled deletion phase can outlive both the shutdown grace
            # period and asyncio.run()'s executor join -- one directory can hold
            # many aged temp files (an agent workspace's temp/output) and unlink
            # latency on network storage is tens of milliseconds. Chunked rather
            # than per-victim so a stop mid-deletion does not discard a typical
            # directory's few victims, which nothing reclaims until next startup.
            if (
                stop_event is not None
                and victims_seen
                and victims_seen % _TEMP_CLEANUP_STOP_CHECK_ENTRIES == 0
                and stop_event.is_set()
            ):
                stopped_early = True
                break
            try:
                os.unlink(victim)
                cleaned_count += 1
                logger.debug("Cleaned up orphaned temp file: %s", victim)
            except OSError as e:
                logger.warning(
                    "Failed to clean up orphaned temp file %s: %s", victim, e
                )

    if stopped_early:
        # WHY: an interrupted sweep must not look identical to a completed one
        # in the logs -- operators need to know the tree was not fully walked.
        # This is the authoritative signal; the background wrapper in app.py
        # deliberately does not re-derive it from the stop flag.
        logger.info(
            "Orphaned temp-file sweep stopped early by shutdown signal after "
            "removing %d file(s); the uploads tree was not fully walked",
            cleaned_count,
        )
    elif cleaned_count > 0:
        logger.info("Cleaned up %d orphaned temporary file(s)", cleaned_count)

    return cleaned_count
