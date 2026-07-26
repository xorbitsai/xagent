"""Durable-file ownership regressions for task execution finalization."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.file_storage.storage import FsspecFileStorage
from xagent.web.api import websocket as websocket_api
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.task_lease_service import (
    TaskLease,
    TaskLeaseHeartbeatOutcome,
)
from xagent.web.services.uploaded_file_store import (
    UploadedFileStore,
    UploadedFileVersionConflict,
)

from .conftest import (
    _direct_db_session,
    _install_one_slot_queue_pool,
)

pytestmark = pytest.mark.usefixtures("_test_db")


def _seed_running_task(*, runner_id: str, run_id: str) -> tuple[int, int]:
    db = _direct_db_session()
    try:
        user = User(
            username=f"output-owner-{run_id}",
            password_hash="hash",
            is_admin=False,
        )
        db.add(user)
        db.flush()
        task = Task(
            user_id=int(user.id),
            title="Durable output",
            status=TaskStatus.RUNNING,
            runner_id=runner_id,
            run_id=run_id,
        )
        db.add(task)
        db.commit()
        return int(task.id), int(user.id)
    finally:
        db.close()


def test_output_prepare_excludes_compensating_metadata() -> None:
    task_id, user_id = _seed_running_task(
        runner_id="compensating-runner",
        run_id="compensating-run",
    )
    db = _direct_db_session()
    try:
        db.add(
            UploadedFile(
                file_id="compensating-output",
                user_id=user_id,
                task_id=task_id,
                filename="compensating.txt",
                storage_path="/tmp/compensating.txt",
                storage_key=(
                    f"users/{user_id}/tasks/{task_id}/outputs/"
                    "compensating-output/compensating.txt"
                ),
                storage_status="compensating",
                mime_type="text/plain",
                file_size=12,
            )
        )
        db.commit()
    finally:
        db.close()

    prepared = websocket_api._prepare_task_file_outputs_isolated(
        task_id=task_id,
        task_user_id=user_id,
        file_outputs=[{"file_id": "compensating-output"}],
        resolved_scope_segments=(),
    )

    assert prepared.normalized_outputs == ()
    assert prepared.mutations == ()


@pytest.mark.asyncio
async def test_output_path_resolution_releases_pool_between_multiple_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id, user_id = _seed_running_task(
        runner_id="path-runner",
        run_id="path-run",
    )
    uploads_dir = tmp_path / "uploads"
    output_dir = uploads_dir / f"user_{user_id}" / f"web_task_{task_id}" / "output"
    output_dir.mkdir(parents=True)
    output_paths = [output_dir / "first.txt", output_dir / "second.txt"]
    for path in output_paths:
        path.write_text(path.name, encoding="utf-8")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.5)

    second_resolution_started = threading.Event()
    allow_second_resolution = threading.Event()
    original_resolve = websocket_api._resolve_output_storage_path
    resolution_count = 0

    def gated_resolve(raw_path: str):
        nonlocal resolution_count
        resolution_count += 1
        resolved = original_resolve(raw_path)
        if resolution_count == 2:
            second_resolution_started.set()
            assert allow_second_resolution.wait(timeout=3)
        return resolved

    monkeypatch.setattr(
        websocket_api,
        "_resolve_output_storage_path",
        gated_resolve,
    )
    prepared: websocket_api._PreparedTaskFileOutputs | None = None
    try:
        worker = asyncio.create_task(
            asyncio.to_thread(
                websocket_api._prepare_task_file_outputs_isolated,
                task_id=task_id,
                task_user_id=user_id,
                file_outputs=[
                    {"path": str(path), "filename": path.name} for path in output_paths
                ],
                resolved_scope_segments=(),
            )
        )
        assert await asyncio.to_thread(second_resolution_started.wait, 2)
        assert engine.pool.checkedout() == 0
        allow_second_resolution.set()
        prepared = await worker
    finally:
        allow_second_resolution.set()
        if prepared is not None:
            websocket_api._settle_prepared_task_file_outputs(
                prepared,
                metadata_committed=False,
            )
        engine.dispose()
        get_unscoped_file_storage.cache_clear()


@pytest.mark.asyncio
async def test_output_path_resolution_releases_pool_after_owner_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id, user_id = _seed_running_task(
        runner_id="owner-runner",
        run_id="owner-run",
    )
    uploads_dir = tmp_path / "uploads"
    output_path = (
        uploads_dir / f"user_{user_id}" / f"web_task_{task_id}" / "output" / "owner.txt"
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_text("owner", encoding="utf-8")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.5)

    resolution_started = threading.Event()
    allow_resolution = threading.Event()
    original_resolve = websocket_api._resolve_output_storage_path

    def gated_resolve(raw_path: str):
        resolved = original_resolve(raw_path)
        resolution_started.set()
        assert allow_resolution.wait(timeout=3)
        return resolved

    monkeypatch.setattr(
        websocket_api,
        "_resolve_output_storage_path",
        gated_resolve,
    )
    prepared: websocket_api._PreparedTaskFileOutputs | None = None
    try:
        worker = asyncio.create_task(
            asyncio.to_thread(
                websocket_api._prepare_task_file_outputs_isolated,
                task_id=task_id,
                task_user_id=None,
                file_outputs=[{"path": str(output_path), "filename": output_path.name}],
                resolved_scope_segments=(),
            )
        )
        assert await asyncio.to_thread(resolution_started.wait, 2)
        assert engine.pool.checkedout() == 0
        allow_resolution.set()
        prepared = await worker
    finally:
        allow_resolution.set()
        if prepared is not None:
            websocket_api._settle_prepared_task_file_outputs(
                prepared,
                metadata_committed=False,
            )
        engine.dispose()
        get_unscoped_file_storage.cache_clear()


@pytest.mark.asyncio
async def test_output_staging_holds_no_pool_slot_or_task_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id, user_id = _seed_running_task(runner_id="runner-a", run_id="run-a")
    uploads_dir = tmp_path / "uploads"
    output_path = (
        uploads_dir / f"user_{user_id}" / f"web_task_{task_id}" / "output" / "a.txt"
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_text("durable output", encoding="utf-8")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.5)

    put_started = threading.Event()
    allow_put = threading.Event()
    original_put_file = FsspecFileStorage.put_file

    def blocking_put_file(
        self: FsspecFileStorage,
        source: Path,
        key: str,
        content_type: str | None = None,
    ):
        assert engine.pool.checkedout() == 0
        put_started.set()
        assert allow_put.wait(timeout=3)
        return original_put_file(self, source, key, content_type)

    monkeypatch.setattr(FsspecFileStorage, "put_file", blocking_put_file)
    try:
        staging = asyncio.create_task(
            asyncio.to_thread(
                websocket_api._prepare_task_file_outputs_isolated,
                task_id=task_id,
                task_user_id=user_id,
                file_outputs=[{"path": str(output_path), "filename": "a.txt"}],
                resolved_scope_segments=(),
            )
        )
        assert await asyncio.to_thread(put_started.wait, 2)
        assert engine.pool.checkedout() == 0

        def read_and_lock_task() -> tuple[str, str]:
            db = _direct_db_session()
            try:
                row = db.query(Task).filter(Task.id == task_id).with_for_update().one()
                return str(row.runner_id), str(row.run_id)
            finally:
                db.close()

        assert await asyncio.to_thread(read_and_lock_task) == ("runner-a", "run-a")
        allow_put.set()
        prepared = await staging
        assert "/_versions/" in prepared.staged_files[0].storage_key

        finalized = websocket_api._finalize_task_execution_result_isolated(
            task_id=task_id,
            task_user_id=user_id,
            pre_run_status=TaskStatus.RUNNING,
            result={"success": True, "output": "done", "file_outputs": []},
            expected_run_id="run-a",
            task_lease=TaskLease(
                task_id=task_id,
                runner_id="runner-a",
                run_id="run-a",
            ),
            resolved_scope_segments=(),
            prepared_outputs=prepared,
        )

        assert not finalized.late_result
        check_db = _direct_db_session()
        try:
            assert (
                check_db.query(UploadedFile)
                .filter(UploadedFile.task_id == task_id)
                .count()
                == 1
            )
        finally:
            check_db.close()
    finally:
        allow_put.set()
        engine.dispose()
        get_unscoped_file_storage.cache_clear()


@pytest.mark.asyncio
async def test_takeover_during_output_upload_cannot_commit_old_run_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id, user_id = _seed_running_task(runner_id="runner-old", run_id="run-old")
    uploads_dir = tmp_path / "uploads"
    output_path = (
        uploads_dir / f"user_{user_id}" / f"web_task_{task_id}" / "output" / "stale.txt"
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_text("stale output", encoding="utf-8")
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()
    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.5)

    put_started = threading.Event()
    allow_put = threading.Event()
    original_put_file = FsspecFileStorage.put_file

    def blocking_put_file(
        self: FsspecFileStorage,
        source: Path,
        key: str,
        content_type: str | None = None,
    ):
        assert engine.pool.checkedout() == 0
        put_started.set()
        assert allow_put.wait(timeout=3)
        return original_put_file(self, source, key, content_type)

    monkeypatch.setattr(FsspecFileStorage, "put_file", blocking_put_file)
    try:
        staging = asyncio.create_task(
            asyncio.to_thread(
                websocket_api._prepare_task_file_outputs_isolated,
                task_id=task_id,
                task_user_id=user_id,
                file_outputs=[{"path": str(output_path), "filename": "stale.txt"}],
                resolved_scope_segments=(),
            )
        )
        assert await asyncio.to_thread(put_started.wait, 2)

        takeover_db = _direct_db_session()
        try:
            task = takeover_db.query(Task).filter(Task.id == task_id).one()
            task.runner_id = "runner-new"
            task.run_id = "run-new"
            takeover_db.commit()
        finally:
            takeover_db.close()

        allow_put.set()
        prepared = await staging
        finalized = websocket_api._finalize_task_execution_result_isolated(
            task_id=task_id,
            task_user_id=user_id,
            pre_run_status=TaskStatus.RUNNING,
            result={"success": True, "output": "stale", "file_outputs": []},
            expected_run_id="run-old",
            task_lease=TaskLease(
                task_id=task_id,
                runner_id="runner-old",
                run_id="run-old",
            ),
            resolved_scope_segments=(),
            prepared_outputs=prepared,
        )

        assert finalized.late_result
        check_db = _direct_db_session()
        try:
            assert check_db.query(UploadedFile).count() == 0
            current = check_db.query(Task).filter(Task.id == task_id).one()
            assert (current.runner_id, current.run_id) == ("runner-new", "run-new")
        finally:
            check_db.close()
        assert not any(path.is_file() for path in object_root.rglob("*"))
    finally:
        allow_put.set()
        engine.dispose()
        get_unscoped_file_storage.cache_clear()


def test_metadata_version_change_rolls_back_entire_task_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id, user_id = _seed_running_task(
        runner_id="cas-runner",
        run_id="cas-run",
    )
    uploads_dir = tmp_path / "uploads"
    output_path = (
        uploads_dir / f"user_{user_id}" / f"web_task_{task_id}" / "output" / "cas.txt"
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_text("old bytes", encoding="utf-8")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    storage = get_unscoped_file_storage()
    try:
        db = _direct_db_session()
        try:
            existing = UploadedFileStore(db).create_from_local_path(
                local_path=output_path,
                user_id=user_id,
                task_id=task_id,
                file_id="cas-output",
                filename="cas.txt",
                storage_key=(
                    f"users/{user_id}/tasks/{task_id}/outputs/cas-output/"
                    "_versions/11111111111111111111111111111111/output/cas.txt"
                ),
                workspace_relative_path="output/cas.txt",
                workspace_category="output",
            )
            original_key = str(existing.storage_key)
            db.commit()
        finally:
            db.close()

        output_path.write_text("stale finalizer bytes", encoding="utf-8")
        prepared = websocket_api._prepare_task_file_outputs_isolated(
            task_id=task_id,
            task_user_id=user_id,
            file_outputs=[{"path": str(output_path), "filename": "cas.txt"}],
            resolved_scope_segments=(),
        )
        staged_key = prepared.staged_files[0].storage_key
        assert storage.exists(staged_key)

        external_key = (
            f"users/{user_id}/tasks/{task_id}/outputs/cas-output/"
            "_versions/22222222222222222222222222222222/output/cas.txt"
        )
        concurrent_db = _direct_db_session()
        try:
            concurrent_record = (
                concurrent_db.query(UploadedFile)
                .filter(UploadedFile.file_id == "cas-output")
                .one()
            )
            concurrent_record.storage_key = external_key
            concurrent_record.checksum = "external-checksum"
            concurrent_record.etag = "external-etag"
            concurrent_db.commit()
        finally:
            concurrent_db.close()

        with pytest.raises(UploadedFileVersionConflict):
            websocket_api._finalize_task_execution_result_isolated(
                task_id=task_id,
                task_user_id=user_id,
                pre_run_status=TaskStatus.RUNNING,
                result={
                    "success": True,
                    "output": "must not commit",
                    "file_outputs": [],
                },
                expected_run_id="cas-run",
                task_lease=TaskLease(
                    task_id=task_id,
                    runner_id="cas-runner",
                    run_id="cas-run",
                ),
                resolved_scope_segments=(),
                prepared_outputs=prepared,
            )

        check_db = _direct_db_session()
        try:
            current = (
                check_db.query(UploadedFile)
                .filter(UploadedFile.file_id == "cas-output")
                .one()
            )
            task = check_db.query(Task).filter(Task.id == task_id).one()
            assert current.storage_key == external_key
            assert current.checksum == "external-checksum"
            assert task.status == TaskStatus.RUNNING
            assert task.output is None
        finally:
            check_db.close()
        assert not storage.exists(staged_key)
        assert storage.exists(original_key)
    finally:
        get_unscoped_file_storage.cache_clear()


def test_resumed_finalizer_uses_same_prepared_output_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id, user_id = _seed_running_task(
        runner_id="resume-runner",
        run_id="resume-run",
    )
    uploads_dir = tmp_path / "uploads"
    output_path = (
        uploads_dir
        / f"user_{user_id}"
        / f"web_task_{task_id}"
        / "output"
        / "resume.txt"
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_text("resume output", encoding="utf-8")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    try:
        prepared = websocket_api._prepare_task_file_outputs_isolated(
            task_id=task_id,
            task_user_id=user_id,
            file_outputs=[{"path": str(output_path), "filename": "resume.txt"}],
            resolved_scope_segments=(),
        )
        finalized = websocket_api._finalize_resumed_task(
            task_id,
            status="completed",
            success=True,
            output="resume done",
            task_owner_user_id=user_id,
            result={"success": True, "output": "resume done", "file_outputs": []},
            task_lease=TaskLease(
                task_id=task_id,
                runner_id="resume-runner",
                run_id="resume-run",
            ),
            prepared_outputs=prepared,
        )

        assert not finalized["late_result"]
        assert finalized["final_status"] == TaskStatus.COMPLETED.value
        db = _direct_db_session()
        try:
            output = (
                db.query(UploadedFile).filter(UploadedFile.task_id == task_id).one()
            )
            assert output.filename == "resume.txt"
            task = db.query(Task).filter(Task.id == task_id).one()
            assert task.status == TaskStatus.COMPLETED
        finally:
            db.close()
    finally:
        get_unscoped_file_storage.cache_clear()


def test_lost_resume_owner_compensates_new_version_without_deleting_committed_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id, user_id = _seed_running_task(
        runner_id="version-runner-old",
        run_id="version-run-old",
    )
    uploads_dir = tmp_path / "uploads"
    output_path = (
        uploads_dir
        / f"user_{user_id}"
        / f"web_task_{task_id}"
        / "output"
        / "versioned.txt"
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_text("committed bytes", encoding="utf-8")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    try:
        db = _direct_db_session()
        try:
            existing = UploadedFileStore(db).create_from_local_path(
                local_path=output_path,
                user_id=user_id,
                task_id=task_id,
                file_id="committed-output",
                filename="versioned.txt",
                storage_key=(
                    f"users/{user_id}/tasks/{task_id}/outputs/"
                    "committed-output/output/versioned.txt"
                ),
                workspace_relative_path="output/versioned.txt",
                workspace_category="output",
            )
            committed_key = str(existing.storage_key)
            db.commit()
        finally:
            db.close()

        output_path.write_text("stale replacement", encoding="utf-8")
        prepared = websocket_api._prepare_task_file_outputs_isolated(
            task_id=task_id,
            task_user_id=user_id,
            file_outputs=[{"path": str(output_path), "filename": "versioned.txt"}],
            resolved_scope_segments=(),
        )
        assert prepared.staged_files[0].storage_key != committed_key

        takeover_db = _direct_db_session()
        try:
            task = takeover_db.query(Task).filter(Task.id == task_id).one()
            task.runner_id = "version-runner-new"
            task.run_id = "version-run-new"
            takeover_db.commit()
        finally:
            takeover_db.close()

        finalized = websocket_api._finalize_resumed_task(
            task_id,
            status="completed",
            success=True,
            output="stale",
            task_owner_user_id=user_id,
            result={"success": True, "output": "stale", "file_outputs": []},
            task_lease=TaskLease(
                task_id=task_id,
                runner_id="version-runner-old",
                run_id="version-run-old",
            ),
            prepared_outputs=prepared,
        )

        assert finalized["late_result"]
        check_db = _direct_db_session()
        try:
            record = (
                check_db.query(UploadedFile)
                .filter(UploadedFile.file_id == "committed-output")
                .one()
            )
            assert record.storage_key == committed_key
        finally:
            check_db.close()
        with get_unscoped_file_storage().open_read(committed_key) as handle:
            assert handle.read() == b"committed bytes"
        assert not get_unscoped_file_storage().exists(
            prepared.staged_files[0].storage_key
        )
    finally:
        get_unscoped_file_storage.cache_clear()


def test_superseded_output_is_deleted_only_after_exact_metadata_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id, user_id = _seed_running_task(
        runner_id="commit-runner",
        run_id="commit-run",
    )
    uploads_dir = tmp_path / "uploads"
    output_path = (
        uploads_dir
        / f"user_{user_id}"
        / f"web_task_{task_id}"
        / "output"
        / "committed.txt"
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_text("old bytes", encoding="utf-8")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    try:
        db = _direct_db_session()
        try:
            existing = UploadedFileStore(db).create_from_local_path(
                local_path=output_path,
                user_id=user_id,
                task_id=task_id,
                file_id="replace-after-commit",
                filename="committed.txt",
                storage_key=(
                    f"users/{user_id}/tasks/{task_id}/outputs/"
                    "replace-after-commit/_versions/"
                    "0123456789abcdef0123456789abcdef/output/committed.txt"
                ),
                workspace_relative_path="output/committed.txt",
                workspace_category="output",
            )
            old_key = str(existing.storage_key)
            db.commit()
        finally:
            db.close()

        output_path.write_text("new bytes", encoding="utf-8")
        prepared = websocket_api._prepare_task_file_outputs_isolated(
            task_id=task_id,
            task_user_id=user_id,
            file_outputs=[{"path": str(output_path), "filename": "committed.txt"}],
            resolved_scope_segments=(),
        )
        new_key = prepared.staged_files[0].storage_key
        storage = get_unscoped_file_storage()
        assert storage.exists(old_key)
        assert storage.exists(new_key)

        original_cleanup = websocket_api.cleanup_superseded_uploaded_file_objects
        cleanup_observations: list[tuple[str, TaskStatus]] = []

        def observe_committed_metadata(claims):  # type: ignore[no-untyped-def]
            check_db = _direct_db_session()
            try:
                record = (
                    check_db.query(UploadedFile)
                    .filter(UploadedFile.file_id == "replace-after-commit")
                    .one()
                )
                task = check_db.query(Task).filter(Task.id == task_id).one()
                cleanup_observations.append(
                    (str(record.storage_key), TaskStatus(task.status))
                )
            finally:
                check_db.close()
            return original_cleanup(claims)

        monkeypatch.setattr(
            websocket_api,
            "cleanup_superseded_uploaded_file_objects",
            observe_committed_metadata,
        )
        finalized = websocket_api._finalize_task_execution_result_isolated(
            task_id=task_id,
            task_user_id=user_id,
            pre_run_status=TaskStatus.RUNNING,
            result={"success": True, "output": "done", "file_outputs": []},
            expected_run_id="commit-run",
            task_lease=TaskLease(
                task_id=task_id,
                runner_id="commit-runner",
                run_id="commit-run",
            ),
            resolved_scope_segments=(),
            prepared_outputs=prepared,
        )

        assert not finalized.late_result
        assert cleanup_observations == [(new_key, TaskStatus.COMPLETED)]
        assert not storage.exists(old_key)
        assert storage.exists(new_key)
    finally:
        get_unscoped_file_storage.cache_clear()


def _assert_cleanup_failure_does_not_reclassify_committed_finalization(
    *,
    resume: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_kind = "resume" if resume else "background"
    runner_id = f"{run_kind}-cleanup-runner"
    run_id = f"{run_kind}-cleanup-run"
    task_id, user_id = _seed_running_task(runner_id=runner_id, run_id=run_id)
    uploads_dir = tmp_path / "uploads"
    output_path = (
        uploads_dir
        / f"user_{user_id}"
        / f"web_task_{task_id}"
        / "output"
        / f"{run_kind}.txt"
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_text("old bytes", encoding="utf-8")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    storage = get_unscoped_file_storage()
    try:
        db = _direct_db_session()
        try:
            existing = UploadedFileStore(db).create_from_local_path(
                local_path=output_path,
                user_id=user_id,
                task_id=task_id,
                file_id=f"{run_kind}-cleanup-output",
                filename=output_path.name,
                storage_key=(
                    f"users/{user_id}/tasks/{task_id}/outputs/"
                    f"{run_kind}-cleanup-output/_versions/"
                    f"11111111111111111111111111111111/output/{output_path.name}"
                ),
                workspace_relative_path=f"output/{output_path.name}",
                workspace_category="output",
            )
            old_key = str(existing.storage_key)
            db.commit()
        finally:
            db.close()

        output_path.write_text("new bytes", encoding="utf-8")
        prepared = websocket_api._prepare_task_file_outputs_isolated(
            task_id=task_id,
            task_user_id=user_id,
            file_outputs=[{"path": str(output_path), "filename": output_path.name}],
            resolved_scope_segments=(),
        )
        new_key = prepared.staged_files[0].storage_key
        assert storage.exists(old_key)
        assert storage.exists(new_key)

        cleanup_attempts = 0

        def fail_cleanup_after_commit(claims):  # type: ignore[no-untyped-def]
            nonlocal cleanup_attempts
            cleanup_attempts += 1
            assert claims
            check_db = _direct_db_session()
            try:
                record = (
                    check_db.query(UploadedFile)
                    .filter(UploadedFile.file_id == f"{run_kind}-cleanup-output")
                    .one()
                )
                task = check_db.query(Task).filter(Task.id == task_id).one()
                assert record.storage_key == new_key
                assert task.status == TaskStatus.COMPLETED
            finally:
                check_db.close()
            assert storage.exists(new_key)
            raise RuntimeError("cleanup reference query unavailable")

        monkeypatch.setattr(
            websocket_api,
            "cleanup_superseded_uploaded_file_objects",
            fail_cleanup_after_commit,
        )
        lease = TaskLease(
            task_id=task_id,
            runner_id=runner_id,
            run_id=run_id,
        )
        if resume:
            resumed_finalization = websocket_api._finalize_resumed_task(
                task_id,
                status="completed",
                success=True,
                output="done",
                task_owner_user_id=user_id,
                result={"success": True, "output": "done", "file_outputs": []},
                task_lease=lease,
                prepared_outputs=prepared,
            )
            assert resumed_finalization["late_result"] is False
            assert resumed_finalization["final_status"] == TaskStatus.COMPLETED.value
        else:
            background_finalization = (
                websocket_api._finalize_task_execution_result_isolated(
                    task_id=task_id,
                    task_user_id=user_id,
                    pre_run_status=TaskStatus.RUNNING,
                    result={"success": True, "output": "done", "file_outputs": []},
                    expected_run_id=run_id,
                    task_lease=lease,
                    resolved_scope_segments=(),
                    prepared_outputs=prepared,
                )
            )
            assert background_finalization.late_result is False
            assert (
                background_finalization.final_task_status == TaskStatus.COMPLETED.value
            )

        assert cleanup_attempts == 1
        check_db = _direct_db_session()
        try:
            record = (
                check_db.query(UploadedFile)
                .filter(UploadedFile.file_id == f"{run_kind}-cleanup-output")
                .one()
            )
            task = check_db.query(Task).filter(Task.id == task_id).one()
            assert record.storage_key == new_key
            assert task.status == TaskStatus.COMPLETED
        finally:
            check_db.close()
        assert storage.exists(new_key)
        assert storage.exists(old_key)
    finally:
        get_unscoped_file_storage.cache_clear()


def test_background_cleanup_failure_does_not_reclassify_committed_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _assert_cleanup_failure_does_not_reclassify_committed_finalization(
        resume=False,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )


def test_resume_cleanup_failure_does_not_reclassify_committed_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _assert_cleanup_failure_does_not_reclassify_committed_finalization(
        resume=True,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )


@pytest.mark.asyncio
async def test_legacy_preview_staging_releases_request_pool_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id, user_id = _seed_running_task(
        runner_id="legacy-preview-runner",
        run_id="legacy-preview-run",
    )
    uploads_dir = tmp_path / "uploads"
    relative_path = f"user_{user_id}/web_task_{task_id}/output/legacy-preview.txt"
    local_path = uploads_dir / relative_path
    local_path.parent.mkdir(parents=True)
    local_path.write_text("legacy preview", encoding="utf-8")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    engine = _install_one_slot_queue_pool(monkeypatch, pool_timeout=0.5)
    request_db = _direct_db_session()
    put_pool_counts: list[int] = []
    original_put_file = FsspecFileStorage.put_file

    def observe_put_file(
        self: FsspecFileStorage,
        source: Path,
        key: str,
        content_type: str | None = None,
    ) -> Any:
        put_pool_counts.append(getattr(engine.pool, "checkedout")())
        return original_put_file(self, source, key, content_type)

    monkeypatch.setattr(FsspecFileStorage, "put_file", observe_put_file)
    try:
        request_db.query(User).filter(User.id == user_id).one()
        assert getattr(engine.pool, "checkedout")() == 1

        response = await websocket_api.redirect_legacy_preview(
            relative_path,
            request_db,
        )

        assert response.status_code == 307
        assert response.headers["location"].startswith("/api/files/public/preview/")
        assert put_pool_counts == [0]
    finally:
        request_db.close()
        engine.dispose()
        get_unscoped_file_storage.cache_clear()


@pytest.mark.asyncio
async def test_legacy_preview_metadata_failure_compensates_staged_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id, user_id = _seed_running_task(
        runner_id="legacy-failure-runner",
        run_id="legacy-failure-run",
    )
    uploads_dir = tmp_path / "uploads"
    relative_path = f"user_{user_id}/web_task_{task_id}/output/legacy-failure.txt"
    local_path = uploads_dir / relative_path
    local_path.parent.mkdir(parents=True)
    local_path.write_text("must be compensated", encoding="utf-8")
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()
    request_db = _direct_db_session()

    def fail_metadata(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("metadata insert failed")

    monkeypatch.setattr(
        UploadedFileStore,
        "upsert_already_durable",
        fail_metadata,
    )
    try:
        with pytest.raises(RuntimeError, match="metadata insert failed"):
            await websocket_api.redirect_legacy_preview(
                relative_path,
                request_db,
            )

        check_db = _direct_db_session()
        try:
            assert (
                check_db.query(UploadedFile)
                .filter(UploadedFile.storage_path == str(local_path.resolve()))
                .count()
                == 0
            )
        finally:
            check_db.close()
        assert not any(path.is_file() for path in object_root.rglob("*"))
    finally:
        request_db.close()
        get_unscoped_file_storage.cache_clear()


@pytest.mark.asyncio
async def test_cancelled_legacy_preview_drains_failed_registration_compensation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_id, user_id = _seed_running_task(
        runner_id="legacy-cancel-runner",
        run_id="legacy-cancel-run",
    )
    uploads_dir = tmp_path / "uploads"
    relative_path = f"user_{user_id}/web_task_{task_id}/output/legacy-cancel.txt"
    local_path = uploads_dir / relative_path
    local_path.parent.mkdir(parents=True)
    local_path.write_text("cancelled preview", encoding="utf-8")
    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads_dir))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()
    request_db = _direct_db_session()
    metadata_started = threading.Event()
    allow_metadata_failure = threading.Event()

    def fail_metadata_after_cancel(*_args: Any, **_kwargs: Any) -> Any:
        metadata_started.set()
        assert allow_metadata_failure.wait(timeout=3)
        raise RuntimeError("metadata failed after caller cancellation")

    monkeypatch.setattr(
        UploadedFileStore,
        "upsert_already_durable",
        fail_metadata_after_cancel,
    )
    registration = asyncio.create_task(
        websocket_api.redirect_legacy_preview(relative_path, request_db)
    )
    try:
        assert await asyncio.to_thread(metadata_started.wait, 2)
        registration.cancel()
        await asyncio.sleep(0)
        assert not registration.done()

        allow_metadata_failure.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(registration, timeout=3)

        check_db = _direct_db_session()
        try:
            assert (
                check_db.query(UploadedFile)
                .filter(UploadedFile.storage_path == str(local_path.resolve()))
                .count()
                == 0
            )
        finally:
            check_db.close()
        assert not any(path.is_file() for path in object_root.rglob("*"))
    finally:
        allow_metadata_failure.set()
        request_db.close()
        get_unscoped_file_storage.cache_clear()


@pytest.mark.asyncio
async def test_cancelled_output_staging_compensates_late_result_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller cancellation must not orphan objects produced by the worker."""

    stage_started = threading.Event()
    allow_stage = threading.Event()
    compensation_finished = threading.Event()
    prepared = websocket_api._PreparedTaskFileOutputs((), (), ())

    def gated_stage(**_kwargs: Any) -> websocket_api._PreparedTaskFileOutputs:
        stage_started.set()
        assert allow_stage.wait(timeout=3)
        return prepared

    def record_compensation(
        actual: websocket_api._PreparedTaskFileOutputs,
        *,
        metadata_committed: bool,
    ) -> None:
        assert actual is prepared
        assert metadata_committed is False
        compensation_finished.set()

    monkeypatch.setattr(
        websocket_api,
        "_prepare_task_file_outputs_isolated",
        gated_stage,
    )
    monkeypatch.setattr(
        websocket_api,
        "_settle_prepared_task_file_outputs",
        record_compensation,
    )

    staging = asyncio.create_task(
        websocket_api._prepare_task_file_outputs_cancellation_safe(
            task_id=42,
            task_user_id=1,
            file_outputs=[{"path": "/slow/output.txt"}],
            resolved_scope_segments=(),
        )
    )
    assert await asyncio.to_thread(stage_started.wait, 2)
    staging.cancel()
    await asyncio.sleep(0)
    assert not staging.done()

    allow_stage.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(staging, timeout=3)

    assert compensation_finished.is_set()


@pytest.mark.asyncio
async def test_resume_output_staging_finishes_while_heartbeat_is_still_active() -> None:
    lease = TaskLease(task_id=42, runner_id="resume-runner", run_id="resume-run")
    heartbeat_stop = asyncio.Event()
    stage_started = threading.Event()
    allow_stage = threading.Event()
    prepared = websocket_api._PreparedTaskFileOutputs((), (), ())

    async def heartbeat() -> TaskLeaseHeartbeatOutcome:
        await heartbeat_stop.wait()
        return TaskLeaseHeartbeatOutcome()

    heartbeat_task = asyncio.create_task(heartbeat())

    class FakeTracker:
        quota_interrupt_reason = None

        def __init__(self, *, task_id: int, **kwargs: Any) -> None:
            assert task_id == 42
            assert kwargs == {
                "expected_run_id": "resume-run",
                "expected_runner_id": "resume-runner",
            }

        async def start_tracking(self) -> None:
            return None

        async def complete_tracking(self) -> None:
            return None

        async def interrupt_reason_for_quota(self) -> None:
            return None

    def gated_stage(**_kwargs: Any) -> websocket_api._PreparedTaskFileOutputs:
        assert not heartbeat_stop.is_set()
        assert not heartbeat_task.done()
        stage_started.set()
        assert allow_stage.wait(timeout=3)
        assert not heartbeat_stop.is_set()
        assert not heartbeat_task.done()
        return prepared

    def finalize(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["prepared_outputs"] is prepared
        assert heartbeat_stop.is_set()
        assert heartbeat_task.done()
        return {
            "task_title": "resume",
            "task_description": "",
            "task_execution_mode": "flash",
            "task_agent_id": None,
            "agent_name": None,
            "agent_logo_url": None,
            "final_status": TaskStatus.COMPLETED.value,
            "lease_released": True,
            "control_event_state": {},
            "normalized_outputs": [],
            "output": "done",
            "late_result": False,
        }

    agent_service = MagicMock()
    agent_service.resume_execution_by_id = AsyncMock(
        return_value={
            "status": "completed",
            "success": True,
            "output": "done",
            "file_outputs": [{"path": "/slow/output.txt"}],
        }
    )

    with (
        patch(
            "xagent.web.api.websocket._prepare_task_file_outputs_isolated",
            side_effect=gated_stage,
        ),
        patch(
            "xagent.web.api.websocket._finalize_resumed_task",
            side_effect=finalize,
        ),
        patch(
            "xagent.web.api.websocket.manager",
            MagicMock(broadcast_to_task=AsyncMock()),
        ),
        patch("xagent.web.api.websocket.background_task_manager.promote_resume_task"),
        patch("xagent.web.api.websocket.background_task_manager.cleanup_task"),
        patch("xagent.web.tracking.task_tracker.TaskTracker", FakeTracker),
    ):
        resume_task = asyncio.create_task(
            websocket_api.execute_resume_background(
                task_id=42,
                agent_service=agent_service,
                task_owner_user_id=1,
                expected_run_id="resume-run",
                resolved_execution_scope=None,
                preacquired_lease=lease,
                preacquired_heartbeat_stop=heartbeat_stop,
                preacquired_heartbeat_task=heartbeat_task,
            )
        )
        assert await asyncio.to_thread(stage_started.wait, 2)
        assert not heartbeat_stop.is_set()
        assert not heartbeat_task.done()
        allow_stage.set()
        await asyncio.wait_for(resume_task, timeout=3)
