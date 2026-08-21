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
