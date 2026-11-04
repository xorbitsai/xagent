"""Bounded disposal retains the production claim across worker activations."""

# ruff: noqa: F401, F811

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, get_ident

import pytest
import sqlalchemy as sa
from filelock import FileLock, Timeout

from tests.web.services.test_complete_upload_cleanup import (
    claim,
    collect,
    completed,
    lifecycle,
    recover,
)
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services import uploaded_file_store as store


def recover_until_deleted(lifecycle, limit=40):
    for _ in range(limit):
        result = recover(lifecycle)
        assert result.failed == 0
        if result.deleted:
            completed(lifecycle)
            return
    raise AssertionError("Recovery did not converge")


def create_converter_tree(previews, file_id):
    converter = previews[0].parent / f".{file_id}.preview-abandoned"
    converter.mkdir()
    for index in range(400):
        (converter / f"partial-{index}.pdf").touch()
    return converter


def test_registered_compensation_hands_off_budget_yield_to_recovery(lifecycle):
    sessions, _, _, previews, key, file_id = lifecycle
    converter = create_converter_tree(previews, file_id)

    store.compensate_registered_uploads_sync(
        [store.RegisteredUploadCompensationClaim(1, file_id, None, key)]
    )
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.storage_status == "compensating"
        assert "previews" not in row.cleanup_manifest["done"]
    recover_until_deleted(lifecycle)
    assert not converter.exists()


@pytest.mark.parametrize("shape", ["files", "directories"])
def test_converter_disposal_obeys_budgets_and_survives_shared_namespace_mutation(
    lifecycle, monkeypatch, shape
):
    sessions, source, _, previews, _, file_id = lifecycle
    if shape == "files":
        converter = create_converter_tree(previews, file_id)
    else:
        converter = previews[0].parent / f".{file_id}.preview-abandoned"
        leaf = converter
        for _ in range(32):
            leaf /= "prefix"
        leaf.mkdir(parents=True)
        for index in range(200):
            (leaf / f"empty-{index}").mkdir()
    nested = converter / "nested"
    nested.mkdir()
    external = previews[0].parent.parent / "keep.txt"
    external.write_text("unrelated")
    (nested / "external-link").symlink_to(external)
    converter_inodes = {
        (info.st_dev, info.st_ino)
        for info in (
            path.stat()
            for path in [converter, *converter.rglob("*")]
            if path.is_dir() and not path.is_symlink()
        )
    }
    for index in range(600):
        (previews[0].parent / f"unrelated-{index}").touch()
        (source.parent / f"unrelated-{index}").touch()
    deleted = entries = 0
    original_unlink, original_rmdir, original_scandir = os.unlink, os.rmdir, os.scandir

    def unlink(*args, **kwargs):
        nonlocal deleted
        deleted += 1
        return original_unlink(*args, **kwargs)

    def rmdir(*args, **kwargs):
        nonlocal deleted
        deleted += 1
        return original_rmdir(*args, **kwargs)

    class CountedScan:
        def __init__(self, scan):
            self.scan = scan

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.scan.close()

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal entries
            entries += 1
            return next(self.scan)

    def scandir(path):
        scan = original_scandir(path)
        if isinstance(path, int):
            info = os.fstat(path)
            if (info.st_dev, info.st_ino) in converter_inodes:
                return CountedScan(scan)
        return scan

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Unbounded recursive disposal")

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "rmdir", rmdir)
    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(shutil, "rmtree", forbidden)
    assert collect(lifecycle).deleted == 0
    assert deleted <= 128 and entries <= 256
    if shape == "directories":
        assert entries == 256 and deleted < 128
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.cleanup_manifest["done"] == ["durable", "local"]
        resource = next(
            r for r in row.cleanup_manifest["previews"] if r["path"] == str(converter)
        )
        assert (converter.parent / resource["quarantine"]).is_dir()
    for tick in range(12):
        (previews[0].parent / f"other-owner-{tick}").touch()
        deleted = entries = 0
        result = recover(lifecycle)
        assert deleted <= 128 and entries <= 256
        assert result.failed == 0
        if result.deleted:
            break
    else:
        raise AssertionError("Shared namespace changes starved disposal")
    completed(lifecycle)
    assert external.read_text() == "unrelated"
    assert len(list(previews[0].parent.glob("unrelated-*"))) == 600


def test_each_activation_in_a_fresh_process_finishes_retained_disposal(lifecycle):
    sessions, _, _, previews, _, file_id = lifecycle
    converter = create_converter_tree(previews, file_id)
    for index in range(600):
        (previews[0].parent / f"unrelated-{index}").touch()
    assert collect(lifecycle).deleted == 0
    engine = sessions.kw["bind"]
    url = str(engine.url)
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            name = conn.exec_driver_sql("SELECT current_database()").scalar_one()
        url = (
            sa.engine.make_url(os.environ["XAGENT_TEST_POSTGRES_URL"])
            .set(database=name)
            .render_as_string(hide_password=False)
        )
    script = """
from datetime import UTC, datetime, timedelta
from xagent.web.models.database import configure_db, get_session_local
from xagent.web.services.uploaded_file_recovery import recover_stale_uploaded_file_compensations_batch_isolated
configure_db()
result = recover_stale_uploaded_file_compensations_batch_isolated(
    session_factory=get_session_local(), cutoff=datetime.now(UTC) + timedelta(days=1), batch_size=10)
assert result.failed == 0
"""
    for tick in range(12):
        (previews[0].parent / f"other-owner-{tick}").touch()
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=dict(os.environ, DATABASE_URL=url),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        with sessions() as db:
            if db.query(UploadedFile).count() == 0:
                break
    else:
        raise AssertionError("Fresh workers did not finish bounded disposal")
    completed(lifecycle)
    assert not converter.exists()


def test_failed_budget_progress_commit_retains_quarantine_and_recovers(lifecycle):
    sessions, _, _, previews, _, file_id = lifecycle
    converter = create_converter_tree(previews, file_id)
    engine = sessions.kw["bind"]
    failed = False

    def fail_progress(_connection, _cursor, statement, *_args):
        nonlocal failed
        if not statement.startswith("UPDATE uploaded_files") or failed:
            return
        with os.scandir(converter.parent) as directories:
            quarantines = [
                Path(entry.path)
                for entry in directories
                if entry.name.startswith(".cleanup-")
                and entry.is_dir(follow_symlinks=False)
            ]
        if any(0 < len(list(path.iterdir())) < 400 for path in quarantines):
            failed = True
            raise sa.exc.OperationalError(
                statement, (), RuntimeError("budget progress commit interrupted")
            )

    sa.event.listen(engine, "before_cursor_execute", fail_progress)
    try:
        assert collect(lifecycle).deleted == 0
    finally:
        sa.event.remove(engine, "before_cursor_execute", fail_progress)
    assert failed
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.storage_status == "compensating"
        assert "previews" not in row.cleanup_manifest["done"]
    recover_until_deleted(lifecycle)
    assert not converter.exists()


def test_converter_depth_limit_preserves_unfinished_obligation(lifecycle):
    sessions, _, _, previews, _, file_id = lifecycle
    converter = previews[0].parent / f".{file_id}.preview-abandoned"
    leaf = converter
    for _ in range(64):
        leaf /= "nested"
    leaf.mkdir(parents=True)
    (leaf / "keep").write_text("retained")
    assert collect(lifecycle).deleted == 0
    assert recover(lifecycle).failed == 1
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert "previews" not in row.cleanup_manifest["done"]
        resource = next(
            r for r in row.cleanup_manifest["previews"] if r["path"] == str(converter)
        )
    retained_leaf = converter.parent / resource["quarantine"]
    for _ in range(64):
        retained_leaf /= "nested"
    assert (retained_leaf / "keep").read_text() == "retained"


@pytest.mark.parametrize("stage", ["open", "finish"])
def test_converter_replacement_preserves_both_directories(
    lifecycle, monkeypatch, stage
):
    sessions, _, _, previews, _, file_id = lifecycle
    converter = previews[0].parent / f".{file_id}.preview-abandoned"
    converter.mkdir()
    (converter / "original").touch()
    inode = converter.stat().st_ino
    original_open, original_stat, original_unlink = os.open, os.stat, os.unlink
    swapped = False
    drained = False
    original_copy = converter.parent / "original-converter"
    replacement = None

    def swap(name, parent):
        nonlocal swapped, replacement
        os.rename(name, original_copy.name, src_dir_fd=parent, dst_dir_fd=parent)
        os.mkdir(name, dir_fd=parent)
        replacement = converter.parent / name
        (replacement / "keep").touch()
        swapped = True

    def open_directory(path, flags, *args, **kwargs):
        parent = kwargs.get("dir_fd")
        if (
            stage == "open"
            and not swapped
            and parent is not None
            and flags & os.O_DIRECTORY
            and str(path).startswith(".cleanup-")
            and original_stat(path, dir_fd=parent).st_ino == inode
        ):
            swap(path, parent)
        return original_open(path, flags, *args, **kwargs)

    def unlink(path, *args, **kwargs):
        nonlocal drained
        result = original_unlink(path, *args, **kwargs)
        parent = kwargs.get("dir_fd")
        if (
            path == "original"
            and parent is not None
            and os.fstat(parent).st_ino == inode
        ):
            drained = True
        return result

    def inspect(path, *args, **kwargs):
        info = original_stat(path, *args, **kwargs)
        parent = kwargs.get("dir_fd")
        if (
            stage == "finish"
            and drained
            and not swapped
            and parent is not None
            and stat.S_ISDIR(info.st_mode)
            and info.st_ino == inode
            and str(path).startswith(".cleanup-")
        ):
            swap(path, parent)
            return original_stat(path, *args, **kwargs)
        return info

    monkeypatch.setattr(os, "open", open_directory)
    monkeypatch.setattr(os, "stat", inspect)
    monkeypatch.setattr(os, "unlink", unlink)
    assert collect(lifecycle).deleted == 0
    assert swapped
    assert replacement is not None and (replacement / "keep").exists()
    assert original_copy.is_dir()
    if stage == "open":
        assert (original_copy / "original").exists()
    with sessions() as db:
        assert "previews" not in db.query(UploadedFile).one().cleanup_manifest["done"]


@pytest.mark.parametrize(
    "done", [[], ["durable"], ["durable", "local"], ["durable", "local", "previews"]]
)
def test_real_v1_captured_temps_preserve_receipts(lifecycle, done):
    sessions, source, _, previews, key, file_id = lifecycle
    prefix = hashlib.sha256(file_id.encode()).hexdigest()[:24]
    temps = []
    for index in range(3):
        path = source.parent / f".{prefix}.{source.name}.{index}.tmp"
        path.touch()
        temps.append(path)
    claim(lifecycle)
    with sessions.begin() as db:
        row = db.query(UploadedFile).one()
        manifest = dict(row.cleanup_manifest)
        assert manifest["version"] == 1 and "discovery" not in manifest
        assert sum(r["path"].endswith(".tmp") for r in manifest["local"]) == 3
        if "durable" in done:
            from xagent.core.file_storage.factory import get_unscoped_file_storage

            get_unscoped_file_storage().delete(key)
        if "local" in done:
            for resource in manifest["local"]:
                Path(resource["path"]).unlink(missing_ok=True)
        if "previews" in done:
            for preview in previews:
                preview.unlink()
        manifest["done"] = done
        row.cleanup_manifest = manifest
        row.updated_at = datetime.now(UTC)
    if "local" in done:
        for path in temps:
            path.write_text("replacement after local receipt")
    assert recover(lifecycle).deleted == 1
    completed(lifecycle)
    if "local" in done:
        assert all(
            path.read_text() == "replacement after local receipt" for path in temps
        )
    else:
        assert not any(path.exists() for path in temps)


def test_unknown_manifest_version_is_retained_without_reinterpreting_evidence(
    lifecycle,
):
    sessions, source, _, previews, _, _ = lifecycle
    claim(lifecycle)
    with sessions.begin() as db:
        row = db.query(UploadedFile).one()
        manifest = dict(row.cleanup_manifest)
        manifest["version"] = 2
        manifest["discovery"] = {"local": {"complete": False}}
        row.cleanup_manifest = manifest
        row.updated_at = datetime.now(UTC)
    assert recover(lifecycle).failed == 1
    with sessions() as db:
        assert db.query(UploadedFile).one().cleanup_manifest == manifest
    assert source.exists() and all(path.exists() for path in previews)


def test_many_captured_local_resources_resume_without_early_receipt(lifecycle):
    sessions, source, _, _, _, file_id = lifecycle
    prefix = hashlib.sha256(file_id.encode()).hexdigest()[:24]
    temps = []
    for index in range(350):
        path = source.parent / f".{prefix}.{source.name}.{index}.tmp"
        path.touch()
        temps.append(path)
    assert collect(lifecycle).deleted == 0
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.cleanup_manifest["done"] == ["durable"]
        assert row.cleanup_manifest["disposal_positions"]["local"] == 128
    recover_until_deleted(lifecycle)
    assert not any(path.exists() for path in temps)


def test_overlapping_workers_wait_for_disposal_and_keep_sql_reference_locks_free(
    lifecycle, monkeypatch
):
    from xagent.web.services.uploaded_file_cleanup import run_uploaded_file_cleanup

    sessions, _, _, previews, key, file_id = lifecycle
    converter = create_converter_tree(previews, file_id)
    inode = converter.stat().st_ino
    candidate, token = claim(lifecycle)
    entered, proceed, contended = Event(), Event(), Event()
    original = os.scandir
    original_acquire = FileLock.acquire
    second_thread = None
    engine = sessions.kw["bind"]
    checked_out = 0

    def checkout(*_args):
        nonlocal checked_out
        checked_out += 1

    def returned(*_args):
        nonlocal checked_out
        checked_out -= 1

    def acquire(lock, *args, **kwargs):
        if get_ident() == second_thread and lock.lock_file.endswith(".cleanup.lock"):
            try:
                with original_acquire(lock, timeout=0):
                    raise AssertionError(
                        "Second worker acquired the active cleanup lock"
                    )
            except Timeout:
                contended.set()
        return original_acquire(lock, *args, **kwargs)

    def blocked(path):
        if (
            isinstance(path, int)
            and os.fstat(path).st_ino == inode
            and not entered.is_set()
        ):
            assert checked_out == 0
            lock = (
                Path(os.environ["LANCEDB_DIR"])
                / ".file-references"
                / (hashlib.sha256(file_id.encode()).hexdigest() + ".lock")
            )
            with FileLock(lock, timeout=0):
                entered.set()
                assert proceed.wait(10)
        return original(path)

    monkeypatch.setattr(os, "scandir", blocked)
    monkeypatch.setattr(FileLock, "acquire", acquire)
    sa.event.listen(engine, "checkout", checkout)
    sa.event.listen(engine, "checkin", returned)  # codespell:ignore checkin

    def first():
        return run_uploaded_file_cleanup(
            session_factory=sessions,
            row_id=candidate.row_id,
            user_id=candidate.user_id,
            file_id=file_id,
            task_id=None,
            storage_key=key,
            expected_updated_at=token,
            compensation_delete=store.delete_uploaded_file_compensation_object,
        )

    def second():
        nonlocal second_thread
        second_thread = get_ident()
        return recover(lifecycle)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_worker = pool.submit(first)
            try:
                if not entered.wait(10):
                    first_worker.result(timeout=10)
                    raise AssertionError("First worker did not enter disposal")
                second_worker = pool.submit(second)
                assert contended.wait(10)
                assert not second_worker.done()
            finally:
                proceed.set()
            assert first_worker.result(timeout=10) == "yielded"
            assert second_worker.result(timeout=10).deferred_budget == 1
        assert first() == "stale"
        recover_until_deleted(lifecycle)
    finally:
        sa.event.remove(engine, "checkout", checkout)
        sa.event.remove(engine, "checkin", returned)  # codespell:ignore checkin


@pytest.mark.asyncio
async def test_cancelled_collector_drains_disposal_and_preserves_recovery(
    lifecycle, monkeypatch
):
    from xagent.web.services.db_runtime import run_db_io_cancellation_safe
    from xagent.web.services.orphan_upload_gc import (
        sweep_orphaned_taskless_uploads_isolated,
    )

    sessions, _, _, previews, _, file_id = lifecycle
    converter = create_converter_tree(previews, file_id)
    inode = converter.stat().st_ino
    entered, proceed = Event(), Event()
    original = os.scandir

    def blocked(path):
        if (
            isinstance(path, int)
            and os.fstat(path).st_ino == inode
            and not entered.is_set()
        ):
            entered.set()
            assert proceed.wait(10)
        return original(path)

    monkeypatch.setattr(os, "scandir", blocked)
    task = asyncio.create_task(
        run_db_io_cancellation_safe(
            lambda: sweep_orphaned_taskless_uploads_isolated(
                session_factory=sessions, older_than_seconds=1
            )
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.storage_status == "compensating"
        assert "previews" not in row.cleanup_manifest["done"]
    recover_until_deleted(lifecycle)
