import os
import threading
import time
from pathlib import Path

from xagent.web.api.kb import cleanup_orphaned_temp_files

# Any file whose mtime is older than this is treated as orphaned by the sweep.
ORPHAN_AGE_SECONDS = 3600


def _write_aged(path: Path, *, age: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))


def test_cleanup_removes_only_aged_orphan_temp_files_across_subtree(tmp_path: Path):
    old = ORPHAN_AGE_SECONDS + 600

    # Orphaned temp files (old enough) — should be removed, including nested.
    _write_aged(tmp_path / "old.tmp-replace", age=old)
    _write_aged(tmp_path / "data.ab12cd.tmp", age=old)
    _write_aged(tmp_path / "sub" / "nested.tmp-replace", age=old)
    _write_aged(tmp_path / "sub" / "deep" / "d.xy.tmp", age=old)

    # Kept: recent orphan, non-temp file, and the two-dot-part `report.tmp`
    # (only "report" + "tmp", so it is not a NamedTemporaryFile-style name).
    _write_aged(tmp_path / "fresh.ab12cd.tmp", age=0)
    _write_aged(tmp_path / "keep.txt", age=old)
    _write_aged(tmp_path / "report.tmp", age=old)

    removed = cleanup_orphaned_temp_files(upload_dir=tmp_path)

    assert removed == 4
    assert not (tmp_path / "old.tmp-replace").exists()
    assert not (tmp_path / "data.ab12cd.tmp").exists()
    assert not (tmp_path / "sub" / "nested.tmp-replace").exists()
    assert not (tmp_path / "sub" / "deep" / "d.xy.tmp").exists()
    assert (tmp_path / "fresh.ab12cd.tmp").exists()
    assert (tmp_path / "keep.txt").exists()
    assert (tmp_path / "report.tmp").exists()


def test_cleanup_returns_zero_for_missing_dir(tmp_path: Path):
    assert cleanup_orphaned_temp_files(upload_dir=tmp_path / "does-not-exist") == 0


def test_cleanup_stops_early_when_stop_event_is_set(tmp_path: Path):
    _write_aged(tmp_path / "old.tmp-replace", age=ORPHAN_AGE_SECONDS + 600)

    stop = threading.Event()
    stop.set()

    removed = cleanup_orphaned_temp_files(upload_dir=tmp_path, stop_event=stop)

    assert removed == 0
    assert (tmp_path / "old.tmp-replace").exists()


class _StopAfterFirstPoll:
    """Stop flag that is unset on its first poll and set on every poll after.

    The seed (root) directory is always popped on the first poll, so the walk
    processes root and then breaks before descending into any subdirectory.
    This pins the per-directory granularity without depending on the exact poll
    count or on scandir ordering: if the flag were checked only once before the
    loop it would be unset (first poll), so the subdir file would be swept too.
    """

    def __init__(self) -> None:
        self._polled = False

    def is_set(self) -> bool:
        if not self._polled:
            self._polled = True
            return False
        return True


def test_cleanup_stops_midwalk_at_directory_boundary(tmp_path: Path):
    old = ORPHAN_AGE_SECONDS + 600
    # One orphan in the root, one nested a level down. The stop trips after the
    # root directory is processed, so the subdir is never descended into.
    _write_aged(tmp_path / "root_orphan.tmp-replace", age=old)
    _write_aged(tmp_path / "sub" / "nested.tmp-replace", age=old)

    stop = _StopAfterFirstPoll()

    removed = cleanup_orphaned_temp_files(upload_dir=tmp_path, stop_event=stop)

    assert removed == 1
    assert not (tmp_path / "root_orphan.tmp-replace").exists()
    assert (tmp_path / "sub" / "nested.tmp-replace").exists()


def test_cleanup_excludes_symlinks_named_like_temp_files(tmp_path: Path):
    # Agent file registration creates real symlinks named like temp files under
    # the uploads tree (websocket.py _register_uploaded_files_for_agent). Those
    # point into live workspaces and must be skipped, not unlinked. The old
    # os.walk-based sweep would stat through the link and remove it.
    target = tmp_path / "workspace_input" / "payload.bin"
    _write_aged(target, age=ORPHAN_AGE_SECONDS + 600)
    link = tmp_path / "data.v1.tmp"
    link.symlink_to(target)

    removed = cleanup_orphaned_temp_files(upload_dir=tmp_path)

    assert removed == 0
    assert link.is_symlink()
    assert target.exists()


def test_cleanup_skips_directory_that_fails_to_open(tmp_path, monkeypatch):
    old = ORPHAN_AGE_SECONDS + 600
    _write_aged(tmp_path / "good" / "keep.tmp-replace", age=old)
    (tmp_path / "bad").mkdir()

    import xagent.web.api.kb as kb_module

    real_scandir = os.scandir

    def _flaky_scandir(path, *args, **kwargs):
        if str(path).endswith("bad"):
            raise OSError("stale handle")
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(kb_module.os, "scandir", _flaky_scandir)

    removed = cleanup_orphaned_temp_files(upload_dir=tmp_path)

    assert removed == 1
    assert not (tmp_path / "good" / "keep.tmp-replace").exists()


def test_cleanup_tolerates_error_mid_iteration(tmp_path, monkeypatch):
    # A directory-level OSError raised while iterating entries (e.g. NFS ESTALE)
    # must skip that directory, not abort the whole sweep.
    old = ORPHAN_AGE_SECONDS + 600
    _write_aged(tmp_path / "root_orphan.tmp-replace", age=old)

    import xagent.web.api.kb as kb_module

    real_scandir = os.scandir

    class _RaiseAtEnd:
        """Yields the real entries, then raises OSError instead of stopping."""

        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._inner.close()
            return False

        def __iter__(self):
            return self

        def __next__(self):
            try:
                return next(self._inner)
            except StopIteration:
                raise OSError("ESTALE mid-iteration") from None

    def _wrapped_scandir(path, *args, **kwargs):
        return _RaiseAtEnd(real_scandir(path, *args, **kwargs))

    monkeypatch.setattr(kb_module.os, "scandir", _wrapped_scandir)

    removed = cleanup_orphaned_temp_files(upload_dir=tmp_path)

    # The orphan was unlinked before the mid-iteration error, and the sweep
    # returned normally instead of propagating the OSError.
    assert removed == 1
    assert not (tmp_path / "root_orphan.tmp-replace").exists()


def test_cleanup_tolerates_unlink_failure_and_keeps_sweeping(tmp_path, monkeypatch):
    old = ORPHAN_AGE_SECONDS + 600
    _write_aged(tmp_path / "boom.tmp-replace", age=old)
    _write_aged(tmp_path / "ok.tmp-replace", age=old)

    import xagent.web.api.kb as kb_module

    real_unlink = os.unlink

    def _selective_unlink(path, *args, **kwargs):
        if str(path).endswith("boom.tmp-replace"):
            raise OSError("permission denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(kb_module.os, "unlink", _selective_unlink)

    removed = cleanup_orphaned_temp_files(upload_dir=tmp_path)

    assert removed == 1
    assert (tmp_path / "boom.tmp-replace").exists()
    assert not (tmp_path / "ok.tmp-replace").exists()
