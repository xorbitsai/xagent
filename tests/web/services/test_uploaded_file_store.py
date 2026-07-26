from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.execution_scope import ExecutionScope
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.file_storage.storage import FsspecFileStorage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.file_turn import bind_turn_files_no_commit
from xagent.web.services.managed_file_ref import DurableStorageOperationError
from xagent.web.services.uploaded_file_store import (
    AppliedUploadedFileVersion,
    LocalUploadRegistration,
    RegisteredUploadCompensationClaim,
    StagedUploadedFile,
    SupersededObjectCleanupClaim,
    UploadedFileStore,
    UploadedFileVersionConflict,
    cleanup_superseded_uploaded_file_objects,
    compensate_registered_uploads_sync,
    compensate_staged_uploaded_files,
    delete_legacy_preview_caches,
    delete_pptx_pdf_cache,
    delete_svg_png_cache,
    register_local_uploads_sync,
    snapshot_uploaded_file_version,
    stage_uploaded_file_from_local_path,
)


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def _user(db):
    user = User(username="store-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _staged_version(
    *,
    user_id: int,
    file_id: str,
    task_id: int | None,
    storage_path: str,
    storage_key: str,
    checksum: str,
    etag: str | None = None,
) -> StagedUploadedFile:
    return StagedUploadedFile(
        file_id=file_id,
        user_id=user_id,
        task_id=task_id,
        filename=storage_key.rsplit("/", 1)[-1],
        storage_path=storage_path,
        storage_backend="file",
        storage_key=storage_key,
        storage_uri=f"file:///{storage_key}",
        checksum=checksum,
        etag=etag,
        workspace_relative_path=f"output/{storage_key.rsplit('/', 1)[-1]}",
        workspace_category="output",
        mime_type="text/plain",
        file_size=7,
    )


def _generation_key(
    *,
    user_id: int,
    task_id: int,
    file_id: str,
    generation: str,
    filename: str = "result.txt",
) -> str:
    return (
        f"users/{user_id}/tasks/{task_id}/outputs/{file_id}/"
        f"_versions/{generation}/{filename}"
    )


def test_upsert_already_durable_rejects_a_compensating_version():
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    old_key = _generation_key(
        user_id=user_id,
        task_id=9,
        file_id="file-cas",
        generation="1" * 32,
    )
    record = UploadedFile(
        file_id="file-cas",
        user_id=user_id,
        task_id=None,
        filename="result.txt",
        storage_path="/tmp/result.txt",
        storage_backend="file",
        storage_key=old_key,
        checksum="old-checksum",
        storage_status="available",
        file_size=7,
    )
    db.add(record)
    db.commit()
    expected = snapshot_uploaded_file_version(record)

    record.storage_status = "compensating"
    db.commit()
    staged = _staged_version(
        user_id=user_id,
        file_id="file-cas",
        task_id=None,
        storage_path="/tmp/result.txt",
        storage_key=_generation_key(
            user_id=user_id,
            task_id=9,
            file_id="file-cas",
            generation="2" * 32,
        ),
        checksum="new-checksum",
    )

    with pytest.raises(UploadedFileVersionConflict):
        UploadedFileStore(db).upsert_already_durable(staged, expected=expected)

    db.expire_all()
    persisted = db.query(UploadedFile).filter_by(file_id="file-cas").one()
    assert persisted.storage_status == "compensating"
    assert persisted.storage_key == old_key


def test_upsert_already_durable_cas_matches_nullable_version_fields():
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    record = UploadedFile(
        file_id="file-null-cas",
        user_id=user_id,
        task_id=None,
        filename="result.txt",
        storage_path="/tmp/null-result.txt",
        storage_backend=None,
        storage_key=None,
        checksum=None,
        etag=None,
        storage_status="legacy",
        file_size=7,
    )
    db.add(record)
    db.commit()
    expected = snapshot_uploaded_file_version(record)
    new_key = _generation_key(
        user_id=user_id,
        task_id=9,
        file_id="file-null-cas",
        generation="3" * 32,
    )

    applied = UploadedFileStore(db).upsert_already_durable(
        _staged_version(
            user_id=user_id,
            file_id="file-null-cas",
            task_id=None,
            storage_path="/tmp/null-result.txt",
            storage_key=new_key,
            checksum="new-checksum",
        ),
        expected=expected,
    )
    db.commit()

    assert isinstance(applied, AppliedUploadedFileVersion)
    assert applied.snapshot.storage_key == new_key
    assert applied.superseded_cleanup_claim is None


def test_upsert_already_durable_rejects_task_rebind_without_explicit_policy():
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    record = UploadedFile(
        file_id="file-task-rebind-default",
        user_id=user_id,
        task_id=21,
        filename="result.txt",
        storage_path="/tmp/rebind-default.txt",
        storage_backend="file",
        storage_key=_generation_key(
            user_id=user_id,
            task_id=21,
            file_id="file-task-rebind-default",
            generation="a" * 32,
        ),
        checksum="old-checksum",
        storage_status="available",
        file_size=7,
    )
    db.add(record)
    db.commit()
    expected = snapshot_uploaded_file_version(record)

    with pytest.raises(ValueError, match="bound to another task"):
        UploadedFileStore(db).upsert_already_durable(
            _staged_version(
                user_id=user_id,
                file_id="file-task-rebind-default",
                task_id=22,
                storage_path="/tmp/rebind-default.txt",
                storage_key=_generation_key(
                    user_id=user_id,
                    task_id=22,
                    file_id="file-task-rebind-default",
                    generation="b" * 32,
                ),
                checksum="new-checksum",
            ),
            expected=expected,
        )


def test_upsert_already_durable_allows_explicit_same_owner_task_rebind():
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    record = UploadedFile(
        file_id="file-task-rebind",
        user_id=user_id,
        task_id=23,
        filename="result.txt",
        storage_path="/tmp/rebind.txt",
        storage_backend="file",
        storage_key=_generation_key(
            user_id=user_id,
            task_id=23,
            file_id="file-task-rebind",
            generation="c" * 32,
        ),
        checksum="old-checksum",
        storage_status="available",
        file_size=7,
    )
    db.add(record)
    db.commit()
    expected = snapshot_uploaded_file_version(record)
    new_key = _generation_key(
        user_id=user_id,
        task_id=24,
        file_id="file-task-rebind",
        generation="d" * 32,
    )

    UploadedFileStore(db).upsert_already_durable(
        _staged_version(
            user_id=user_id,
            file_id="file-task-rebind",
            task_id=24,
            storage_path="/tmp/rebind.txt",
            storage_key=new_key,
            checksum="new-checksum",
        ),
        expected=expected,
        allow_task_rebind=True,
    )
    db.commit()

    persisted = db.query(UploadedFile).filter_by(file_id="file-task-rebind").one()
    assert persisted.user_id == user_id
    assert persisted.task_id == 24
    assert persisted.storage_key == new_key


def test_explicit_task_rebind_still_rejects_a_stale_version():
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    record = UploadedFile(
        file_id="file-task-rebind-stale",
        user_id=user_id,
        task_id=25,
        filename="result.txt",
        storage_path="/tmp/rebind-stale.txt",
        storage_backend="file",
        storage_key=_generation_key(
            user_id=user_id,
            task_id=25,
            file_id="file-task-rebind-stale",
            generation="e" * 32,
        ),
        checksum="old-checksum",
        storage_status="available",
        file_size=7,
    )
    db.add(record)
    db.commit()
    expected = snapshot_uploaded_file_version(record)
    record.checksum = "concurrent-checksum"
    db.commit()

    with pytest.raises(UploadedFileVersionConflict):
        UploadedFileStore(db).upsert_already_durable(
            _staged_version(
                user_id=user_id,
                file_id="file-task-rebind-stale",
                task_id=26,
                storage_path="/tmp/rebind-stale.txt",
                storage_key=_generation_key(
                    user_id=user_id,
                    task_id=26,
                    file_id="file-task-rebind-stale",
                    generation="f" * 32,
                ),
                checksum="new-checksum",
            ),
            expected=expected,
            allow_task_rebind=True,
        )


def test_explicit_task_rebind_rejects_cross_owner_replacement():
    db = _session()
    owner = _user(db)
    other = User(username="other-store-user", password_hash="hash", is_admin=False)
    db.add(other)
    db.commit()
    db.refresh(other)
    owner_id = int(owner.id)
    other_id = int(other.id)
    record = UploadedFile(
        file_id="file-task-rebind-owner",
        user_id=owner_id,
        task_id=27,
        filename="result.txt",
        storage_path="/tmp/rebind-owner.txt",
        storage_backend="file",
        storage_key=_generation_key(
            user_id=owner_id,
            task_id=27,
            file_id="file-task-rebind-owner",
            generation="1" * 32,
        ),
        checksum="old-checksum",
        storage_status="available",
        file_size=7,
    )
    db.add(record)
    db.commit()
    expected = snapshot_uploaded_file_version(record)

    with pytest.raises(ValueError, match="owned by another user"):
        UploadedFileStore(db).upsert_already_durable(
            _staged_version(
                user_id=other_id,
                file_id="file-task-rebind-owner",
                task_id=28,
                storage_path="/tmp/rebind-owner.txt",
                storage_key=_generation_key(
                    user_id=other_id,
                    task_id=28,
                    file_id="file-task-rebind-owner",
                    generation="2" * 32,
                ),
                checksum="new-checksum",
            ),
            expected=expected,
            allow_task_rebind=True,
        )


def test_two_sessions_cannot_apply_the_same_stale_uploaded_file_snapshot(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'version-cas.sqlite'}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as seed_db:
        user = _user(seed_db)
        user_id = int(user.id)
        old_key = _generation_key(
            user_id=user_id,
            task_id=11,
            file_id="file-stale",
            generation="4" * 32,
        )
        seed_db.add(
            UploadedFile(
                file_id="file-stale",
                user_id=user_id,
                task_id=11,
                filename="result.txt",
                storage_path="/tmp/stale-result.txt",
                storage_backend="file",
                storage_key=old_key,
                checksum="old-checksum",
                etag=None,
                storage_status="available",
                file_size=7,
            )
        )
        seed_db.commit()

    first_db = SessionLocal()
    second_db = SessionLocal()
    try:
        first_expected = snapshot_uploaded_file_version(
            first_db.query(UploadedFile).filter_by(file_id="file-stale").one()
        )
        first_db.commit()
        second_expected = snapshot_uploaded_file_version(
            second_db.query(UploadedFile).filter_by(file_id="file-stale").one()
        )
        second_db.commit()

        first_key = _generation_key(
            user_id=user_id,
            task_id=11,
            file_id="file-stale",
            generation="5" * 32,
        )
        first_applied = UploadedFileStore(first_db).upsert_already_durable(
            _staged_version(
                user_id=user_id,
                file_id="file-stale",
                task_id=11,
                storage_path="/tmp/stale-result.txt",
                storage_key=first_key,
                checksum="first-checksum",
            ),
            expected=first_expected,
        )
        first_db.commit()

        with pytest.raises(UploadedFileVersionConflict):
            UploadedFileStore(second_db).upsert_already_durable(
                _staged_version(
                    user_id=user_id,
                    file_id="file-stale",
                    task_id=11,
                    storage_path="/tmp/stale-result.txt",
                    storage_key=_generation_key(
                        user_id=user_id,
                        task_id=11,
                        file_id="file-stale",
                        generation="6" * 32,
                    ),
                    checksum="second-checksum",
                ),
                expected=second_expected,
            )

        assert first_applied.superseded_cleanup_claim == (
            SupersededObjectCleanupClaim(
                user_id=user_id,
                storage_backend="file",
                storage_key=old_key,
            )
        )
        with SessionLocal() as verify_db:
            persisted = (
                verify_db.query(UploadedFile).filter_by(file_id="file-stale").one()
            )
            assert persisted.storage_key == first_key
            assert persisted.checksum == "first-checksum"
    finally:
        first_db.close()
        second_db.close()
        engine.dispose()


@pytest.mark.parametrize(
    ("field_name", "concurrent_value"),
    [
        ("filename", "concurrent-name.txt"),
        ("storage_uri", "file:///concurrent-object"),
        ("workspace_relative_path", "output/concurrent.txt"),
        ("workspace_category", "concurrent-category"),
        ("mime_type", "application/concurrent"),
        ("file_size", 99),
    ],
)
def test_uploaded_file_cas_fences_every_replaced_metadata_field(
    tmp_path,
    field_name,
    concurrent_value,
):
    engine = create_engine(f"sqlite:///{tmp_path / f'{field_name}-cas.sqlite'}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as seed_db:
        user = _user(seed_db)
        user_id = int(user.id)
        seed_db.add(
            UploadedFile(
                file_id="file-complete-cas",
                user_id=user_id,
                task_id=17,
                filename="original.txt",
                storage_path="/tmp/original.txt",
                storage_backend="file",
                storage_key=_generation_key(
                    user_id=user_id,
                    task_id=17,
                    file_id="file-complete-cas",
                    generation="9" * 32,
                ),
                storage_uri="file:///original-object",
                checksum="original-checksum",
                etag="original-etag",
                workspace_relative_path="output/original.txt",
                workspace_category="output",
                storage_status="available",
                mime_type="text/plain",
                file_size=7,
            )
        )
        seed_db.commit()

    stale_db = SessionLocal()
    concurrent_db = SessionLocal()
    try:
        expected = snapshot_uploaded_file_version(
            stale_db.query(UploadedFile).filter_by(file_id="file-complete-cas").one()
        )
        stale_db.commit()

        concurrent_record = (
            concurrent_db.query(UploadedFile)
            .filter_by(file_id="file-complete-cas")
            .one()
        )
        setattr(concurrent_record, field_name, concurrent_value)
        concurrent_db.commit()

        with pytest.raises(UploadedFileVersionConflict):
            UploadedFileStore(stale_db).upsert_already_durable(
                _staged_version(
                    user_id=user_id,
                    file_id="file-complete-cas",
                    task_id=17,
                    storage_path="/tmp/replacement.txt",
                    storage_key=_generation_key(
                        user_id=user_id,
                        task_id=17,
                        file_id="file-complete-cas",
                        generation="a" * 32,
                    ),
                    checksum="replacement-checksum",
                ),
                expected=expected,
            )

        with SessionLocal() as verify_db:
            persisted = (
                verify_db.query(UploadedFile)
                .filter_by(file_id="file-complete-cas")
                .one()
            )
            assert getattr(persisted, field_name) == concurrent_value
            assert persisted.checksum == "original-checksum"
    finally:
        stale_db.close()
        concurrent_db.close()
        engine.dispose()


def test_superseded_generation_cleanup_skips_a_referenced_key(monkeypatch):
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    key = _generation_key(
        user_id=user_id,
        task_id=12,
        file_id="file-referenced",
        generation="7" * 32,
    )
    db.add(
        UploadedFile(
            file_id="file-referenced",
            user_id=user_id,
            task_id=None,
            filename="result.txt",
            storage_path="/tmp/referenced-result.txt",
            storage_backend="file",
            storage_key=key,
            checksum="checksum",
            storage_status="available",
            file_size=7,
        )
    )
    db.commit()
    SessionLocal = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_user_file_storage",
        lambda _user_id: type(
            "RecordingStorage",
            (),
            {
                "backend": "file",
                "delete": lambda _self, object_key: deleted.append(object_key),
            },
        )(),
    )

    cleanup_superseded_uploaded_file_objects(
        (
            SupersededObjectCleanupClaim(
                user_id=user_id,
                storage_backend="file",
                storage_key=key,
            ),
        )
    )

    assert deleted == []


def test_superseded_cleanup_reference_ignores_backend_metadata(monkeypatch):
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    key = _generation_key(
        user_id=user_id,
        task_id=12,
        file_id="file-backend-drift",
        generation="b" * 32,
    )
    db.add(
        UploadedFile(
            file_id="file-backend-drift",
            user_id=user_id,
            task_id=12,
            filename="result.txt",
            storage_path="/tmp/backend-drift.txt",
            storage_backend=None,
            storage_key=key,
            checksum="checksum",
            storage_status="available",
            file_size=7,
        )
    )
    db.commit()
    SessionLocal = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_user_file_storage",
        lambda _user_id: type(
            "RecordingStorage",
            (),
            {
                "backend": "file",
                "delete": lambda _self, object_key: deleted.append(object_key),
            },
        )(),
    )

    failures = cleanup_superseded_uploaded_file_objects(
        (
            SupersededObjectCleanupClaim(
                user_id=user_id,
                storage_backend="file",
                storage_key=key,
            ),
        )
    )

    assert failures == ()
    assert deleted == []


def test_superseded_cleanup_query_failure_is_deferred(monkeypatch):
    claim = SupersededObjectCleanupClaim(
        user_id=1,
        storage_backend="file",
        storage_key=_generation_key(
            user_id=1,
            task_id=13,
            file_id="file-query-failure",
            generation="c" * 32,
        ),
    )

    def fail_session_factory():
        raise RuntimeError("reference lookup unavailable")

    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        fail_session_factory,
    )
    delete_called = False

    def unexpected_storage(_user_id):
        nonlocal delete_called
        delete_called = True
        raise AssertionError("cleanup must not delete after an unknown reference check")

    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_user_file_storage",
        unexpected_storage,
    )

    assert cleanup_superseded_uploaded_file_objects((claim,)) == (claim,)
    assert delete_called is False


def test_staged_compensation_preserves_a_committed_reference(monkeypatch):
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    key = _generation_key(
        user_id=user_id,
        task_id=14,
        file_id="file-commit-ack",
        generation="d" * 32,
    )
    staged = _staged_version(
        user_id=user_id,
        file_id="file-commit-ack",
        task_id=14,
        storage_path="/tmp/commit-ack.txt",
        storage_key=key,
        checksum="committed-checksum",
    )
    db.add(staged.to_record())
    db.commit()
    SessionLocal = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_user_file_storage",
        lambda _user_id: type(
            "RecordingStorage",
            (),
            {
                "delete": lambda _self, object_key: deleted.append(object_key),
            },
        )(),
    )

    assert compensate_staged_uploaded_files((staged,)) == ()
    assert deleted == []


def test_staged_compensation_query_failure_preserves_the_object(monkeypatch):
    staged = _staged_version(
        user_id=1,
        file_id="file-unknown-commit",
        task_id=14,
        storage_path="/tmp/unknown-commit.txt",
        storage_key=_generation_key(
            user_id=1,
            task_id=14,
            file_id="file-unknown-commit",
            generation="e" * 32,
        ),
        checksum="unknown-checksum",
    )

    def fail_session_factory():
        raise RuntimeError("reference lookup unavailable")

    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        fail_session_factory,
    )
    delete_called = False

    def unexpected_storage(_user_id):
        nonlocal delete_called
        delete_called = True
        raise AssertionError("unknown commit state must retain the object")

    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_user_file_storage",
        unexpected_storage,
    )

    assert compensate_staged_uploaded_files((staged,)) == ("file-unknown-commit",)
    assert delete_called is False


def test_immutable_stage_removes_unreferenced_object_after_post_put_failure(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    SessionLocal = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )
    source = tmp_path / "uploads" / "post-put.txt"
    source.parent.mkdir()
    source.write_text("post-put bytes", encoding="utf-8")
    file_id = str(uuid4())
    storage_key = f"users/{user_id}/uploads/{file_id}/post-put.txt"

    def fail_after_bytes_are_written(
        _storage: FsspecFileStorage,
        _key: str,
        _checksum: str,
        *,
        content_type: str | None,
    ) -> None:
        del content_type
        raise RuntimeError("metadata acknowledgement lost")

    monkeypatch.setattr(
        FsspecFileStorage,
        "_store_content_hash",
        fail_after_bytes_are_written,
    )

    with pytest.raises(DurableStorageOperationError):
        stage_uploaded_file_from_local_path(
            local_path=source,
            user_id=user_id,
            file_id=file_id,
            filename=source.name,
            storage_key=storage_key,
        )

    assert get_unscoped_file_storage().exists(storage_key) is False
    get_unscoped_file_storage.cache_clear()


def test_immutable_stage_retains_possible_object_when_reference_state_is_unknown(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    source = tmp_path / "uploads" / "post-put-unknown.txt"
    source.parent.mkdir()
    source.write_text("post-put bytes", encoding="utf-8")
    file_id = str(uuid4())
    storage_key = f"users/1/uploads/{file_id}/post-put-unknown.txt"

    def fail_after_bytes_are_written(
        _storage: FsspecFileStorage,
        _key: str,
        _checksum: str,
        *,
        content_type: str | None,
    ) -> None:
        del content_type
        raise RuntimeError("metadata acknowledgement lost")

    monkeypatch.setattr(
        FsspecFileStorage,
        "_store_content_hash",
        fail_after_bytes_are_written,
    )
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: (_ for _ in ()).throw(RuntimeError("reference lookup unavailable")),
    )

    with pytest.raises(DurableStorageOperationError):
        stage_uploaded_file_from_local_path(
            local_path=source,
            user_id=1,
            file_id=file_id,
            filename=source.name,
            storage_key=storage_key,
        )

    assert get_unscoped_file_storage().exists(storage_key) is True
    get_unscoped_file_storage.cache_clear()


def test_superseded_generation_cleanup_deletes_an_unreferenced_key(monkeypatch):
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    key = _generation_key(
        user_id=user_id,
        task_id=13,
        file_id="file-unreferenced",
        generation="8" * 32,
    )
    SessionLocal = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_user_file_storage",
        lambda _user_id: type(
            "RecordingStorage",
            (),
            {
                "backend": "file",
                "delete": lambda _self, object_key: deleted.append(object_key),
            },
        )(),
    )

    failures = cleanup_superseded_uploaded_file_objects(
        (
            SupersededObjectCleanupClaim(
                user_id=user_id,
                storage_backend="file",
                storage_key=key,
            ),
        )
    )

    assert failures == ()
    assert deleted == [key]


def test_superseded_generation_cleanup_skips_a_legacy_deterministic_key(
    monkeypatch,
):
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    legacy_key = f"users/{user_id}/tasks/14/outputs/file-legacy/output/result.txt"
    SessionLocal = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_user_file_storage",
        lambda _user_id: type(
            "RecordingStorage",
            (),
            {
                "backend": "file",
                "delete": lambda _self, object_key: deleted.append(object_key),
            },
        )(),
    )

    cleanup_superseded_uploaded_file_objects(
        (
            SupersededObjectCleanupClaim(
                user_id=user_id,
                storage_backend="file",
                storage_key=legacy_key,
            ),
        )
    )

    assert deleted == []


def test_create_from_local_path_persists_record_and_syncs_durable(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "input.txt"
    source.parent.mkdir()
    source.write_text("store content", encoding="utf-8")

    record = UploadedFileStore(db).create_from_local_path(
        local_path=source,
        user_id=int(user.id),
        file_id="file-store",
        filename="input.txt",
        mime_type="text/plain",
    )
    db.commit()

    persisted = db.query(UploadedFile).filter_by(file_id="file-store").one()
    assert persisted.id == record.id
    assert persisted.storage_status == "available"
    assert persisted.storage_key == "users/1/uploads/file-store/input.txt"
    with get_unscoped_file_storage().open_read(str(persisted.storage_key)) as handle:
        assert handle.read() == b"store content"


def test_register_failure_compensates_object_in_original_execution_scope(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    SessionLocal = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )

    source = tmp_path / "uploads" / "scoped.txt"
    source.parent.mkdir()
    source.write_text("scoped content", encoding="utf-8")
    scope = ExecutionScope(
        workspace_segments=("clients", "acme"),
        isolate_external_dirs=True,
    )
    file_id = str(uuid4())
    expected_key = f"users/{user_id}/clients/acme/uploads/{file_id}/scoped.txt"
    registration = LocalUploadRegistration(
        local_path=source,
        user_id=user_id,
        file_id=file_id,
        task_id=999,
        filename="scoped.txt",
        mime_type="text/plain",
        execution_scope=scope,
    )

    with pytest.raises(ValueError, match="Task 999 is not owned"):
        register_local_uploads_sync((registration,))

    assert not get_unscoped_file_storage().exists(expected_key)
    assert db.query(UploadedFile).filter_by(file_id=file_id).first() is None


def test_new_pending_sync_rejects_persisted_available_record(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "refresh.txt"
    source.parent.mkdir()
    source.write_text("old content", encoding="utf-8")
    store = UploadedFileStore(db)
    record = store.create_from_local_path(
        local_path=source,
        user_id=int(user.id),
        file_id="file-refresh",
        filename="refresh.txt",
    )
    storage_key = str(record.storage_key)
    db.commit()

    source.write_text("new content", encoding="utf-8")
    with pytest.raises(UploadedFileVersionConflict, match="available"):
        store._sync_new_pending(record)

    with get_unscoped_file_storage().open_read(storage_key) as handle:
        assert handle.read() == b"old content"


def test_delete_removes_local_durable_and_db_record(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "delete.txt"
    source.parent.mkdir()
    source.write_text("delete me", encoding="utf-8")
    store = UploadedFileStore(db)
    record = store.create_from_local_path(
        local_path=source,
        user_id=int(user.id),
        file_id="file-delete",
        filename="delete.txt",
    )
    storage_key = str(record.storage_key)
    assert source.exists()
    assert get_unscoped_file_storage().exists(storage_key)

    store.delete(record, delete_local=True)
    db.commit()

    assert not source.exists()
    assert not get_unscoped_file_storage().exists(storage_key)
    assert db.query(UploadedFile).filter_by(file_id="file-delete").first() is None


def test_delete_svg_png_cache_removes_all_previews_for_file_id(monkeypatch, tmp_path):
    """A file_id can have multiple derived SVG previews cached (e.g. one per
    relative-path asset registered under it). Deleting the source upload must
    remove every one of them, not just a single fixed filename."""

    storage_root = tmp_path / "storage"
    cache_dir = storage_root / "svg_png_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "file-svg.aaa111.preview.png").write_bytes(b"png-a")
    (cache_dir / "file-svg.bbb222.preview.png").write_bytes(b"png-b")
    other_file_cache = cache_dir / "other-file.ccc333.preview.png"
    other_file_cache.write_bytes(b"png-other")

    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(storage_root))

    delete_svg_png_cache("file-svg")

    assert not list(cache_dir.glob("file-svg.*.preview.png"))
    assert other_file_cache.exists()


def test_delete_legacy_preview_caches_removes_path_keyed_previews(
    monkeypatch, tmp_path
):
    """Legacy caches are keyed only by the canonical source path."""
    import hashlib

    storage_root = tmp_path / "storage"
    svg_cache_dir = storage_root / "svg_png_cache"
    pdf_cache_dir = storage_root / "pptx_pdf_cache"
    svg_cache_dir.mkdir(parents=True)
    pdf_cache_dir.mkdir(parents=True)

    legacy_source = tmp_path / "legacy" / "notes.svg"
    legacy_source.parent.mkdir(parents=True)
    legacy_source.write_text("<svg></svg>", encoding="utf-8")

    path_key = hashlib.sha256(str(legacy_source.resolve()).encode()).hexdigest()[:24]
    legacy_svg_cache = svg_cache_dir / f"{path_key}.preview.png"
    legacy_svg_cache.write_bytes(b"legacy-png")
    legacy_pdf_cache = pdf_cache_dir / f"{path_key}.preview.pdf"
    legacy_pdf_cache.write_bytes(b"legacy-pdf")

    unrelated_svg_cache = svg_cache_dir / "some-other-hash.preview.png"
    unrelated_svg_cache.write_bytes(b"unrelated-png")
    unrelated_pdf_cache = pdf_cache_dir / "some-other-hash.preview.pdf"
    unrelated_pdf_cache.write_bytes(b"unrelated-pdf")

    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(storage_root))

    delete_legacy_preview_caches(legacy_source.resolve())

    assert not legacy_svg_cache.exists()
    assert not legacy_pdf_cache.exists()
    assert unrelated_svg_cache.exists()
    assert unrelated_pdf_cache.exists()


def test_delete_svg_png_cache_uses_registered_id_prefix(monkeypatch, tmp_path):
    """Registered cache cleanup needs only the database-owned file ID."""
    storage_root = tmp_path / "storage"
    cache_dir = storage_root / "svg_png_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "file-svg.aaa111.preview.png").write_bytes(b"png-a")

    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(storage_root))

    delete_svg_png_cache("file-svg")

    assert not list(cache_dir.glob("file-svg.*.preview.png"))


def test_delete_svg_png_cache_treats_file_id_as_a_literal_prefix(monkeypatch, tmp_path):
    """A caller-controlled identifier must never be interpreted as a glob."""
    storage_root = tmp_path / "storage"
    cache_dir = storage_root / "svg_png_cache"
    cache_dir.mkdir(parents=True)
    literal_cache = cache_dir / "*.literal.preview.png"
    literal_cache.write_bytes(b"literal")
    unrelated_cache = cache_dir / "other-file.unrelated.preview.png"
    unrelated_cache.write_bytes(b"unrelated")

    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(storage_root))

    delete_svg_png_cache("*")

    assert not literal_cache.exists()
    assert unrelated_cache.exists()


def test_delete_pptx_pdf_cache_rejects_path_segments(monkeypatch, tmp_path):
    storage_root = tmp_path / "storage"
    cache_dir = storage_root / "pptx_pdf_cache"
    cache_dir.mkdir(parents=True)
    outside_cache = storage_root / "target.preview.pdf"
    outside_cache.write_bytes(b"must survive")

    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(storage_root))

    delete_pptx_pdf_cache("../target")

    assert outside_cache.exists()


def test_delete_removes_svg_png_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "delete.svg"
    source.parent.mkdir()
    source.write_text("<svg></svg>", encoding="utf-8")
    store = UploadedFileStore(db)
    record = store.create_from_local_path(
        local_path=source,
        user_id=int(user.id),
        file_id="file-svg-delete",
        filename="delete.svg",
        mime_type="image/svg+xml",
    )

    cache_dir = tmp_path / "storage" / "svg_png_cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "file-svg-delete.deadbeef.preview.png"
    cache_path.write_bytes(b"cached preview")

    store.delete(record, delete_local=True)
    db.commit()

    assert not cache_path.exists()


def test_delete_preserves_db_row_when_durable_cleanup_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "delete-first.txt"
    source.parent.mkdir()
    source.write_text("delete first", encoding="utf-8")
    store = UploadedFileStore(db)
    record = store.create_from_local_path(
        local_path=source,
        user_id=int(user.id),
        file_id="file-delete-first",
        filename="delete-first.txt",
    )
    storage_key = str(record.storage_key)

    from xagent.web.services.managed_file_ref import ManagedFileRef

    def fail_delete_durable(self: ManagedFileRef) -> None:
        raise RuntimeError("simulated durable cleanup failure")

    monkeypatch.setattr(ManagedFileRef, "delete_durable", fail_delete_durable)

    with pytest.raises(RuntimeError, match="simulated durable cleanup failure"):
        store.delete(record, delete_local=True)

    assert (
        db.query(UploadedFile).filter_by(file_id="file-delete-first").first()
        is not None
    )
    assert source.exists()
    assert get_unscoped_file_storage().exists(storage_key)


def test_compensation_preserves_metadata_when_durable_cleanup_fails(
    monkeypatch,
):
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    db.add_all(
        [
            UploadedFile(
                file_id="cleanup-success",
                user_id=user_id,
                filename="success.txt",
                storage_path="/tmp/cleanup-success.txt",
                storage_key="users/1/uploads/cleanup-success/success.txt",
                storage_status="available",
                file_size=7,
            ),
            UploadedFile(
                file_id="cleanup-failure",
                user_id=user_id,
                filename="failure.txt",
                storage_path="/tmp/cleanup-failure.txt",
                storage_key="users/1/uploads/cleanup-failure/failure.txt",
                storage_status="available",
                file_size=7,
            ),
        ]
    )
    db.commit()

    SessionLocal = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )

    class SelectiveFailureStorage:
        def delete(self, key: str) -> None:
            if "cleanup-failure" in key:
                raise RuntimeError("simulated durable cleanup failure")

        def exists(self, key: str) -> bool:
            return "cleanup-failure" in key

    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_user_file_storage",
        lambda _user_id: SelectiveFailureStorage(),
    )

    with pytest.raises(DurableStorageOperationError, match="cleanup-failure"):
        compensate_registered_uploads_sync(
            (
                RegisteredUploadCompensationClaim(
                    user_id=user_id,
                    file_id="cleanup-success",
                    expected_task_id=None,
                    expected_storage_key=(
                        "users/1/uploads/cleanup-success/success.txt"
                    ),
                ),
                RegisteredUploadCompensationClaim(
                    user_id=user_id,
                    file_id="cleanup-failure",
                    expected_task_id=None,
                    expected_storage_key=(
                        "users/1/uploads/cleanup-failure/failure.txt"
                    ),
                ),
            )
        )

    db.expire_all()
    assert db.query(UploadedFile).filter_by(file_id="cleanup-success").first() is None
    failed = db.query(UploadedFile).filter_by(file_id="cleanup-failure").one()
    assert failed.storage_status == "compensating"


def test_compensation_reconciles_delete_ack_loss_as_cleaned(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    file_id = str(uuid4())
    storage_key = f"users/{user_id}/uploads/{file_id}/ack-loss.txt"
    source = tmp_path / "ack-loss.txt"
    source.write_text("payload", encoding="utf-8")
    real_storage = get_unscoped_file_storage()
    real_storage.put_file(source, storage_key)
    db.add(
        UploadedFile(
            file_id=file_id,
            user_id=user_id,
            filename=source.name,
            storage_path=str(source),
            storage_backend="file",
            storage_key=storage_key,
            storage_status="available",
            checksum="checksum",
            file_size=source.stat().st_size,
        )
    )
    db.commit()
    cache_dir = tmp_path / "storage" / "svg_png_cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / f"{file_id}.cached.preview.png"
    cache_path.write_bytes(b"cached preview")
    unrelated_cache = cache_dir / "unrelated.cached.preview.png"
    unrelated_cache.write_bytes(b"unrelated")
    SessionLocal = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )

    class DeleteThenRaise:
        def delete(self, key: str) -> None:
            real_storage.delete(key)
            raise RuntimeError("delete acknowledgement lost")

        def exists(self, key: str) -> bool:
            return real_storage.exists(key)

    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_user_file_storage",
        lambda _user_id: DeleteThenRaise(),
    )

    compensate_registered_uploads_sync(
        (
            RegisteredUploadCompensationClaim(
                user_id=user_id,
                file_id=file_id,
                expected_task_id=None,
                expected_storage_key=storage_key,
            ),
        )
    )

    db.expire_all()
    assert db.query(UploadedFile).filter_by(file_id=file_id).first() is None
    assert real_storage.exists(storage_key) is False
    assert not cache_path.exists()
    assert unrelated_cache.exists()
    get_unscoped_file_storage.cache_clear()


def test_compensation_keeps_unknown_delete_outcome_unavailable(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    file_id = str(uuid4())
    storage_key = f"users/{user_id}/uploads/{file_id}/unknown.txt"
    db.add(
        UploadedFile(
            file_id=file_id,
            user_id=user_id,
            filename="unknown.txt",
            storage_path="/tmp/unknown.txt",
            storage_backend="file",
            storage_key=storage_key,
            storage_status="available",
            checksum="checksum",
            file_size=7,
        )
    )
    db.commit()
    cache_dir = tmp_path / "storage" / "svg_png_cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / f"{file_id}.cached.preview.png"
    cache_path.write_bytes(b"cached preview")
    SessionLocal = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )

    class UnknownDeleteOutcome:
        def delete(self, _key: str) -> None:
            raise RuntimeError("delete acknowledgement lost")

        def exists(self, _key: str) -> bool:
            raise RuntimeError("storage probe unavailable")

    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_user_file_storage",
        lambda _user_id: UnknownDeleteOutcome(),
    )

    with pytest.raises(DurableStorageOperationError, match=file_id):
        compensate_registered_uploads_sync(
            (
                RegisteredUploadCompensationClaim(
                    user_id=user_id,
                    file_id=file_id,
                    expected_task_id=None,
                    expected_storage_key=storage_key,
                ),
            )
        )

    db.expire_all()
    record = db.query(UploadedFile).filter_by(file_id=file_id).one()
    assert record.storage_status == "compensating"
    assert cache_path.exists()


def test_delayed_upload_compensation_does_not_delete_a_file_that_was_bound(
    monkeypatch,
):
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    task = Task(
        user_id=user_id,
        title="consumer won",
        status=TaskStatus.PENDING,
    )
    db.add(task)
    db.flush()
    storage_key = "users/1/uploads/request-file/input.txt"
    db.add(
        UploadedFile(
            file_id="request-file",
            user_id=user_id,
            filename="input.txt",
            storage_path="/tmp/request-file.txt",
            storage_key=storage_key,
            storage_status="available",
            checksum="checksum",
            file_size=7,
        )
    )
    db.commit()

    missing = bind_turn_files_no_commit(
        file_ids=["request-file"],
        task_id=int(task.id),
        owner_user_id=user_id,
        db=db,
    )
    assert missing == []
    db.commit()

    SessionLocal = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )
    deleted_keys: list[str] = []
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_user_file_storage",
        lambda _user_id: type(
            "RecordingStorage",
            (),
            {"delete": lambda _self, key: deleted_keys.append(key)},
        )(),
    )

    compensate_registered_uploads_sync(
        (
            RegisteredUploadCompensationClaim(
                user_id=user_id,
                file_id="request-file",
                expected_task_id=None,
                expected_storage_key=storage_key,
            ),
        )
    )

    db.expire_all()
    persisted = db.query(UploadedFile).filter_by(file_id="request-file").one()
    assert persisted.task_id == task.id
    assert persisted.storage_status == "available"
    assert deleted_keys == []


def test_compensation_claim_blocks_a_concurrent_file_binding(monkeypatch):
    db = _session()
    user = _user(db)
    user_id = int(user.id)
    task = Task(
        user_id=user_id,
        title="compensation won",
        status=TaskStatus.PENDING,
    )
    db.add(task)
    db.flush()
    task_id = int(task.id)
    storage_key = "users/1/uploads/request-file/input.txt"
    db.add(
        UploadedFile(
            file_id="request-file",
            user_id=user_id,
            filename="input.txt",
            storage_path="/tmp/request-file.txt",
            storage_key=storage_key,
            storage_status="available",
            checksum="checksum",
            file_size=7,
        )
    )
    db.commit()

    SessionLocal = sessionmaker(bind=db.get_bind())
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )
    competing_missing: list[str] = []

    class InterleavingStorage:
        def delete(self, key: str) -> None:
            assert key == storage_key
            with SessionLocal() as competing_db:
                competing_missing.extend(
                    bind_turn_files_no_commit(
                        file_ids=["request-file"],
                        task_id=task_id,
                        owner_user_id=user_id,
                        db=competing_db,
                    )
                )
                competing_db.commit()

    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_user_file_storage",
        lambda _user_id: InterleavingStorage(),
    )

    compensate_registered_uploads_sync(
        (
            RegisteredUploadCompensationClaim(
                user_id=user_id,
                file_id="request-file",
                expected_task_id=None,
                expected_storage_key=storage_key,
            ),
        )
    )

    db.expire_all()
    assert competing_missing == ["request-file"]
    assert db.query(UploadedFile).filter_by(file_id="request-file").first() is None


def test_delete_skips_local_file_outside_local_root(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "outside" / "unexpected.txt"
    source.parent.mkdir()
    source.write_text("unexpected", encoding="utf-8")
    store = UploadedFileStore(db)
    record = store.create_from_local_path(
        local_path=source,
        user_id=int(user.id),
        file_id="file-outside",
        filename="unexpected.txt",
    )
    storage_key = str(record.storage_key)
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()

    store.delete(record, delete_local=True, local_root=allowed_root)
    db.commit()

    assert source.exists()
    assert not get_unscoped_file_storage().exists(storage_key)
    assert db.query(UploadedFile).filter_by(file_id="file-outside").first() is None


def test_upsert_by_storage_path_reuses_record_and_refreshes_durable(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "kb.md"
    source.parent.mkdir()
    source.write_text("first", encoding="utf-8")
    store = UploadedFileStore(db)

    first = store.upsert_by_storage_path(
        user_id=int(user.id),
        filename="kb.md",
        storage_path=source,
        mime_type="text/markdown",
        file_size=source.stat().st_size,
    )
    first_key = str(first.storage_key)
    db.commit()

    source.write_text("second", encoding="utf-8")
    second = store.upsert_by_storage_path(
        user_id=int(user.id),
        filename="kb-renamed.md",
        storage_path=source,
        mime_type="text/markdown",
        file_size=source.stat().st_size,
    )
    db.commit()

    assert second.id == first.id
    assert second.filename == "kb-renamed.md"
    assert second.file_size == len("second")
    second_key = str(second.storage_key)
    assert second_key != first_key
    assert not get_unscoped_file_storage().exists(first_key)
    with get_unscoped_file_storage().open_read(second_key) as handle:
        assert handle.read() == b"second"


def test_upsert_by_storage_path_refreshes_same_size_rewrite(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "kb.md"
    source.parent.mkdir()
    source.write_text("old-data", encoding="utf-8")
    store = UploadedFileStore(db)

    first = store.upsert_by_storage_path(
        user_id=int(user.id),
        filename="kb.md",
        storage_path=source,
        mime_type="text/markdown",
        file_size=source.stat().st_size,
    )
    first_key = str(first.storage_key)
    old_checksum = str(first.checksum)
    db.commit()
    cache_dir = tmp_path / "storage" / "svg_png_cache"
    cache_dir.mkdir(parents=True)
    stale_cache = cache_dir / f"{first.file_id}.same-path.preview.png"
    stale_cache.write_bytes(b"old preview")
    unrelated_cache = cache_dir / "unrelated.same-path.preview.png"
    unrelated_cache.write_bytes(b"unrelated")

    source.write_text("new-data", encoding="utf-8")
    second = store.upsert_by_storage_path(
        user_id=int(user.id),
        filename="kb.md",
        storage_path=source,
        mime_type="text/markdown",
        file_size=source.stat().st_size,
    )
    db.commit()

    assert second.id == first.id
    assert second.checksum != old_checksum
    second_key = str(second.storage_key)
    assert second_key != first_key
    assert not get_unscoped_file_storage().exists(first_key)
    assert not stale_cache.exists()
    assert unrelated_cache.exists()
    with get_unscoped_file_storage().open_read(second_key) as handle:
        assert handle.read() == b"new-data"


def test_upsert_by_storage_path_syncs_when_requested_storage_key_changes(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "kb.md"
    source.parent.mkdir()
    source.write_text("same-bytes", encoding="utf-8")
    store = UploadedFileStore(db)
    file_id = str(uuid4())
    first_requested_key = f"users/1/uploads/{file_id}/kb.md"

    first = store.upsert_by_storage_path(
        user_id=int(user.id),
        file_id=file_id,
        filename="kb.md",
        storage_path=source,
        mime_type="text/markdown",
        file_size=source.stat().st_size,
        storage_key=first_requested_key,
    )
    first_key = str(first.storage_key)
    db.commit()

    second_key = (
        f"users/1/uploads/{file_id}/_versions/12345678123456781234567812345678/kb.md"
    )
    second = store.upsert_by_storage_path(
        user_id=int(user.id),
        filename="kb.md",
        storage_path=source,
        mime_type="text/markdown",
        file_size=source.stat().st_size,
        storage_key=second_key,
    )
    db.commit()

    assert second.id == first.id
    assert first_key != second_key
    assert second.storage_key == second_key
    assert not get_unscoped_file_storage().exists(first_key)
    with get_unscoped_file_storage().open_read(second_key) as handle:
        assert handle.read() == b"same-bytes"


def test_upsert_by_storage_path_skips_durable_sync_when_file_unchanged(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "kb.md"
    source.parent.mkdir()
    source.write_text("same", encoding="utf-8")
    store = UploadedFileStore(db)
    first = store.upsert_by_storage_path(
        user_id=int(user.id),
        filename="kb.md",
        storage_path=source,
        mime_type="text/markdown",
        file_size=source.stat().st_size,
    )
    db.commit()

    from xagent.web.services.managed_file_ref import ManagedFileRef

    def fail_sync(self, *, storage_key=None, mime_type=None):
        raise AssertionError("unexpected durable sync for unchanged file")

    monkeypatch.setattr(ManagedFileRef, "sync_to_durable", fail_sync)

    second = store.upsert_by_storage_path(
        user_id=int(user.id),
        filename="kb.md",
        storage_path=source,
        mime_type="text/markdown",
        file_size=source.stat().st_size,
    )

    assert second.id == first.id
    assert second.storage_status == "available"


@pytest.mark.parametrize("storage_status", ["pending", "compensating"])
def test_upsert_by_storage_path_rejects_inflight_states(
    monkeypatch,
    tmp_path,
    storage_status,
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / f"{storage_status}.txt"
    source.parent.mkdir()
    source.write_text("payload", encoding="utf-8")
    db.add(
        UploadedFile(
            file_id=f"{storage_status}-file",
            user_id=int(user.id),
            filename=source.name,
            storage_path=str(source),
            storage_status=storage_status,
            file_size=source.stat().st_size,
        )
    )
    db.commit()

    with pytest.raises(UploadedFileVersionConflict, match=storage_status):
        UploadedFileStore(db).upsert_by_storage_path(
            user_id=int(user.id),
            filename=source.name,
            storage_path=source,
            mime_type="text/plain",
            file_size=source.stat().st_size,
        )

    db.expire_all()
    assert (
        db.query(UploadedFile).filter_by(file_id=f"{storage_status}-file").one()
    ).storage_status == storage_status


def test_upsert_by_storage_path_rejects_another_owner(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    owner = _user(db)
    other = User(username="other-store-user", password_hash="hash", is_admin=False)
    db.add(other)
    db.commit()
    source = tmp_path / "uploads" / "foreign.txt"
    source.parent.mkdir()
    source.write_text("payload", encoding="utf-8")
    db.add(
        UploadedFile(
            file_id="foreign-file",
            user_id=int(owner.id),
            filename=source.name,
            storage_path=str(source),
            storage_status="legacy",
            file_size=source.stat().st_size,
        )
    )
    db.commit()

    with pytest.raises(PermissionError, match="another user"):
        UploadedFileStore(db).upsert_by_storage_path(
            user_id=int(other.id),
            filename=source.name,
            storage_path=source,
            mime_type="text/plain",
            file_size=source.stat().st_size,
        )

    db.expire_all()
    record = db.query(UploadedFile).filter_by(file_id="foreign-file").one()
    assert record.user_id == owner.id
    assert record.storage_status == "legacy"


def test_upsert_by_storage_path_rejects_a_different_file_identity(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "identity.txt"
    source.parent.mkdir()
    source.write_text("payload", encoding="utf-8")
    db.add(
        UploadedFile(
            file_id="existing-file",
            user_id=int(user.id),
            filename=source.name,
            storage_path=str(source),
            storage_status="legacy",
            file_size=source.stat().st_size,
        )
    )
    db.commit()

    with pytest.raises(UploadedFileVersionConflict, match="file_id"):
        UploadedFileStore(db).upsert_by_storage_path(
            user_id=int(user.id),
            file_id="replacement-file",
            filename=source.name,
            storage_path=source,
            mime_type="text/plain",
            file_size=source.stat().st_size,
        )


def test_upsert_by_storage_path_migrates_same_owner_legacy_row(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "legacy.txt"
    source.parent.mkdir()
    source.write_text("payload", encoding="utf-8")
    db.add(
        UploadedFile(
            file_id="legacy-file",
            user_id=int(user.id),
            filename=source.name,
            storage_path=str(source),
            storage_status="legacy",
            file_size=source.stat().st_size,
        )
    )
    db.commit()

    record = UploadedFileStore(db).upsert_by_storage_path(
        user_id=int(user.id),
        filename=source.name,
        storage_path=source,
        mime_type="text/plain",
        file_size=source.stat().st_size,
    )

    assert record.file_id == "legacy-file"
    assert record.storage_status == "available"
    assert "/uploads/legacy-file/_versions/" in str(record.storage_key)


def test_upsert_by_storage_path_rejects_a_stale_expected_version(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "stale.txt"
    source.parent.mkdir()
    source.write_text("original", encoding="utf-8")
    store = UploadedFileStore(db)
    record = store.upsert_by_storage_path(
        user_id=int(user.id),
        filename=source.name,
        storage_path=source,
        mime_type="text/plain",
        file_size=source.stat().st_size,
    )
    db.commit()
    stale = snapshot_uploaded_file_version(record)

    record.filename = "concurrent.txt"
    db.commit()

    with pytest.raises(UploadedFileVersionConflict, match="expected version"):
        store.upsert_by_storage_path(
            user_id=int(user.id),
            file_id=str(record.file_id),
            filename=source.name,
            storage_path=source,
            mime_type="text/plain",
            file_size=source.stat().st_size,
            expected_version=stale,
        )


def test_upsert_by_storage_path_restores_metadata_from_an_exact_receipt(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "restore.txt"
    source.parent.mkdir()
    source.write_text("original", encoding="utf-8")
    store = UploadedFileStore(db)
    original = store.upsert_by_storage_path(
        user_id=int(user.id),
        filename="original.txt",
        storage_path=source,
        mime_type="text/plain",
        file_size=source.stat().st_size,
        workspace_category="source",
    )
    db.commit()
    original_version = snapshot_uploaded_file_version(original)

    source.write_text("replacement", encoding="utf-8")
    replaced = store.upsert_by_storage_path(
        user_id=int(user.id),
        filename="replacement.txt",
        storage_path=source,
        mime_type="text/markdown",
        file_size=source.stat().st_size,
        workspace_category="output",
    )
    db.commit()
    replaced_version = snapshot_uploaded_file_version(replaced)

    source.write_text("original", encoding="utf-8")
    restored = store.upsert_by_storage_path(
        user_id=int(user.id),
        file_id=str(original.file_id),
        filename="ignored-by-snapshot.txt",
        storage_path=source,
        mime_type=None,
        file_size=source.stat().st_size,
        expected_version=replaced_version,
        replacement_metadata=original_version,
    )

    assert restored.filename == "original.txt"
    assert restored.mime_type == "text/plain"
    assert restored.workspace_category == "source"
    assert restored.storage_key != replaced_version.storage_key
    with get_unscoped_file_storage().open_read(str(restored.storage_key)) as handle:
        assert handle.read() == b"original"


def test_create_from_local_path_removes_record_when_durable_write_fails(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    db = _session()
    user = _user(db)
    source = tmp_path / "uploads" / "output.txt"
    source.parent.mkdir()
    source.write_text("output content", encoding="utf-8")

    from xagent.core.file_storage.storage import FsspecFileStorage

    def fail_put_file(self, source, key, content_type=None):
        raise RuntimeError("simulated durable write outage")

    monkeypatch.setattr(FsspecFileStorage, "put_file", fail_put_file)

    with pytest.raises(DurableStorageOperationError):
        UploadedFileStore(db).create_from_local_path(
            local_path=source,
            user_id=int(user.id),
            file_id="file-output",
            filename="output.txt",
            mime_type="text/plain",
        )

    db.commit()
    assert db.query(UploadedFile).filter_by(file_id="file-output").first() is None
