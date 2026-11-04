"""The production collector/recovery lifecycle retains every cleanup obligation."""

from __future__ import annotations

import copy
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.tools.core.RAG_tools.storage.file_reference import file_cleanup_lock
from xagent.web.models import database
from xagent.web.models.database import Base, configure_db, get_engine, get_session_local
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.uploaded_file_cleanup_fence import UploadedFileCleanupFence
from xagent.web.models.user import User
from xagent.web.services import uploaded_file_cleanup as cleanup
from xagent.web.services import uploaded_file_cleanup_resources as resources
from xagent.web.services import uploaded_file_store as store
from xagent.web.services.kb_reference_protection import install_file_reference_validator
from xagent.web.services.managed_file_ref import (
    ManagedFileRef,
)
from xagent.web.services.orphan_upload_gc import (
    TASKLESS_SHARE_UPLOAD_SOURCE,
    _claim_orphan,
    _orphan_candidates,
    cleanup_orphaned_taskless_uploads,
)
from xagent.web.services.uploaded_file_cleanup_publication import (
    FilePublicationUnavailable,
)
from xagent.web.services.uploaded_file_recovery import (
    recover_stale_uploaded_file_compensations_batch_isolated,
)

PAYLOAD = b"owned upload bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def lifecycle(request, tmp_path, monkeypatch):
    from tests.shared.postgres_disposable import disposable_database_factory

    previous = database._SessionLocal, database._engine
    for variable, name in (
        ("XAGENT_UPLOADS_DIR", "uploads"),
        ("XAGENT_FILE_MATERIALIZE_DIR", "materialized"),
        ("XAGENT_STORAGE_ROOT", "storage"),
        ("LANCEDB_DIR", "lance"),
    ):
        monkeypatch.setenv(variable, str(tmp_path / name))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    stack = ExitStack()
    if request.param == "sqlite":
        configure_db(f"sqlite:///{tmp_path / 'metadata.db'}")
        engine = get_engine()
    else:
        make = stack.enter_context(disposable_database_factory("complete_cleanup"))
        engine = make("boundary")
        database._engine = engine
        database._SessionLocal = sessionmaker(bind=engine, autoflush=False)
        install_file_reference_validator()
    Base.metadata.create_all(engine)
    sessions = get_session_local()
    file_id = str(uuid4())
    source = tmp_path / "uploads" / "user_1" / "source.txt"
    source.parent.mkdir(parents=True)
    source.write_bytes(PAYLOAD)
    key = f"users/1/uploads/{file_id}/source.txt"
    durable = get_unscoped_file_storage().put_file(source, key)
    with sessions.begin() as db:
        db.add(User(id=1, username="owner", password_hash="unused"))
        db.flush()
        db.add(
            UploadedFile(
                file_id=file_id,
                user_id=1,
                filename="source.txt",
                storage_path=str(source),
                storage_key=key,
                storage_backend=durable.backend,
                storage_uri=durable.uri,
                storage_status="available",
                checksum=DIGEST,
                file_size=len(PAYLOAD),
                upload_source=TASKLESS_SHARE_UPLOAD_SOURCE,
                created_at=datetime.now(UTC) - timedelta(days=10),
            )
        )
    materialized = get_unscoped_file_storage().materialize(key, "source.txt")
    previews = []
    for directory, filename in (
        ("pptx_pdf_cache", f"{file_id}.preview.pdf"),
        ("svg_png_cache", f"{file_id}.asset.preview.png"),
    ):
        path = tmp_path / "storage" / directory / filename
        path.parent.mkdir(parents=True)
        path.write_bytes(b"derived bytes")
        previews.append(path)
    yield sessions, source, materialized, previews, key, file_id
    engine.dispose()
    database._SessionLocal, database._engine = previous
    get_unscoped_file_storage.cache_clear()
    stack.close()


def collect(lifecycle):
    sessions, *_ = lifecycle
    with sessions() as db:
        return cleanup_orphaned_taskless_uploads(db, older_than_seconds=1)


def claim(lifecycle):
    sessions, *_ = lifecycle
    with sessions() as db:
        candidate = _orphan_candidates(
            db, cutoff=datetime.now(UTC), limit=1, after=None
        )[0]
        token = _claim_orphan(db, candidate)
        assert token is not None
        return candidate, token


def recover(lifecycle):
    return recover_stale_uploaded_file_compensations_batch_isolated(
        session_factory=lifecycle[0],
        cutoff=datetime.now(UTC) + timedelta(days=1),
        batch_size=10,
    )


def retained(lifecycle, done):
    sessions, _, _, _, key, file_id = lifecycle
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.storage_status == "compensating"
        assert row.storage_key == key
        assert row.cleanup_manifest["done"] == done
        assert row.cleanup_manifest["file_id"] == file_id
        return copy.deepcopy(row.cleanup_manifest)


def completed(lifecycle):
    sessions, source, materialized, previews, key, file_id = lifecycle
    with sessions() as db:
        assert db.query(UploadedFile).count() == 0
        assert db.get(UploadedFileCleanupFence, file_id) is not None
    assert not get_unscoped_file_storage().exists(key)
    assert not source.exists()
    assert not materialized.exists()
    assert not any(path.exists() for path in previews)


def test_actual_collector_finishes_every_resource(lifecycle):
    assert collect(lifecycle).deleted == 1
    completed(lifecycle)


@pytest.mark.parametrize("phase", ["durable", "local", "previews", "settlement"])
def test_each_failure_retains_metadata_and_converges_in_fresh_session(
    lifecycle, monkeypatch, phase
):
    sessions, source, materialized, previews, key, _ = lifecycle
    with monkeypatch.context() as failing:
        if phase == "durable":

            def fail_delete(_self, _key):
                raise OSError("transient durable delete")

            failing.setattr(type(get_unscoped_file_storage()), "delete", fail_delete)
        elif phase in {"local", "previews"}:
            original = resources.os.unlink
            preview_directories = {
                (info.st_dev, info.st_ino)
                for preview in previews
                for info in [preview.parent.stat()]
            }

            def fail_unlink(path, *args, **kwargs):
                parent = os.fstat(kwargs["dir_fd"])
                is_preview = (parent.st_dev, parent.st_ino) in preview_directories
                if is_preview == (phase == "previews"):
                    raise OSError("transient unlink")
                return original(path, *args, **kwargs)

            failing.setattr(resources.os, "unlink", fail_unlink)
        else:

            def fail_settlement(*_args, **_kwargs):
                raise OSError("transient settlement")

            failing.setattr(
                store, "settle_uploaded_file_compensation_no_commit", fail_settlement
            )
        assert collect(lifecycle).deleted == 0
    expected = {
        "durable": [],
        "local": ["durable"],
        "previews": ["durable", "local"],
        "settlement": ["durable", "local", "previews"],
    }[phase]
    manifest = retained(lifecycle, expected)
    assert manifest["local"][0]["path"] == str(source)
    assert manifest["local"][1]["path"] == str(materialized)
    assert get_unscoped_file_storage().exists(key) is (phase == "durable")
    assert recover(lifecycle).deleted == 1
    completed(lifecycle)


@pytest.mark.parametrize(
    "done", [[], ["durable"], ["durable", "local"], ["durable", "local", "previews"]]
)
def test_interruption_after_committed_phase_restarts_without_losing_handle(
    lifecycle, monkeypatch, done
):
    original = cleanup._save_manifest

    class Interrupted(BaseException):
        pass

    if not done:
        claim(lifecycle)
    else:

        def interrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            manifest = args[-1]
            if manifest["done"] == done:
                raise Interrupted()
            return result

        with monkeypatch.context() as failing:
            failing.setattr(cleanup, "_save_manifest", interrupt)
            with pytest.raises(Interrupted):
                collect(lifecycle)
    retained(lifecycle, done)
    assert recover(lifecycle).deleted == 1
    completed(lifecycle)


def test_missing_objects_and_copies_are_success(lifecycle):
    _, source, materialized, previews, key, _ = lifecycle
    get_unscoped_file_storage().delete(key)
    for path in (source, materialized, *previews):
        path.unlink()
    assert collect(lifecycle).deleted == 1
    completed(lifecycle)


@pytest.mark.parametrize("presence", ["exists", "unknown"])
def test_non_absent_presence_keeps_all_obligations(lifecycle, presence):
    candidate, token = claim(lifecycle)
    outcome = cleanup.run_uploaded_file_cleanup(
        session_factory=lifecycle[0],
        row_id=candidate.row_id,
        user_id=1,
        file_id=candidate.file_id,
        task_id=None,
        storage_key=candidate.storage_key,
        expected_updated_at=token,
        compensation_delete=lambda **_kwargs: presence,
    )
    assert outcome == presence
    retained(lifecycle, [])
    assert lifecycle[1].exists() and lifecycle[2].exists()
    assert recover(lifecycle).deleted == 1
    completed(lifecycle)


def test_claim_commit_precedes_all_destructive_io_and_returns_connections(
    lifecycle, monkeypatch
):
    sessions, source, *_ = lifecycle
    engine = sessions.kw["bind"]
    active = set()

    def checked_out(connection, _record, _proxy):
        active.add(id(connection))

    def returned(connection, _record):
        active.discard(id(connection))

    sa.event.listen(engine, "checkout", checked_out)
    sa.event.listen(engine, "checkin", returned)  # codespell:ignore checkin
    original_delete = get_unscoped_file_storage().delete
    original_unlink = resources.os.unlink

    def verify():
        assert not active
        with sessions() as db:
            row = db.query(UploadedFile).one()
            assert row.storage_status == "compensating"
            assert row.cleanup_manifest is not None

    def delete(key):
        verify()
        # The reference lock is available even while slow storage is active.
        from xagent.core.tools.core.RAG_tools.storage.file_reference import (
            file_reference_lock,
        )

        with file_reference_lock([lifecycle[-1]]):
            pass
        return original_delete(key)

    def unlink(path, *args, **kwargs):
        verify()
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(get_unscoped_file_storage(), "delete", delete)
    monkeypatch.setattr(resources.os, "unlink", unlink)
    try:
        assert collect(lifecycle).deleted == 1
    finally:
        sa.event.remove(engine, "checkout", checked_out)
        sa.event.remove(engine, "checkin", returned)  # codespell:ignore checkin
    completed(lifecycle)


def test_failed_claim_commit_leaves_local_and_durable_bytes(lifecycle, monkeypatch):
    sessions, source, materialized, previews, key, _ = lifecycle
    with sessions() as db:
        candidate = _orphan_candidates(
            db, cutoff=datetime.now(UTC), limit=1, after=None
        )[0]
        monkeypatch.setattr(
            db, "commit", lambda: (_ for _ in ()).throw(OSError("commit failed"))
        )
        with pytest.raises(OSError):
            _claim_orphan(db, candidate)
        db.rollback()
    assert (
        source.exists()
        and materialized.exists()
        and all(path.exists() for path in previews)
    )
    assert get_unscoped_file_storage().exists(key)
    with sessions() as db:
        assert db.query(UploadedFile).one().storage_status == "available"


@pytest.mark.parametrize(
    "kind", ["external", "shared", "escaping", "replacement", "checksum", "root"]
)
def test_source_boundaries_preserve_unowned_and_uncertain_resources(
    lifecycle, tmp_path, monkeypatch, kind
):
    sessions, source, _, _, _, _ = lifecycle
    retained_path = source
    if kind == "external":
        external = tmp_path / "external.txt"
        external.write_bytes(PAYLOAD)
        with sessions.begin() as db:
            db.query(UploadedFile).update({UploadedFile.storage_path: str(external)})
        source.unlink()
        retained_path = external
    elif kind == "shared":
        retained_path = tmp_path / "shared.txt"
        os.link(source, retained_path)
    elif kind == "escaping":
        outside = tmp_path / "outside.txt"
        outside.write_bytes(PAYLOAD)
        source.unlink()
        source.symlink_to(outside)
        retained_path = outside
    candidate, token = claim(lifecycle)
    if kind == "replacement":
        original = source.with_suffix(".original")
        source.rename(original)
        source.write_bytes(b"new publication")
    elif kind == "checksum":
        source.write_bytes(b"modified managed content")
    elif kind == "root":
        monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "another-root"))
    outcome = cleanup.run_uploaded_file_cleanup(
        session_factory=sessions,
        row_id=candidate.row_id,
        user_id=1,
        file_id=candidate.file_id,
        task_id=None,
        storage_key=candidate.storage_key,
        expected_updated_at=token,
        compensation_delete=store.delete_uploaded_file_compensation_object,
    )
    if kind in {"external", "shared"}:
        assert outcome == "deleted"
        assert retained_path.read_bytes() == PAYLOAD
    else:
        assert outcome == "pending"
        with sessions() as db:
            assert db.query(UploadedFile).one().cleanup_manifest is not None
        assert retained_path.exists()
        if kind == "replacement":
            assert source.read_bytes() == b"new publication"
        elif kind == "root":
            assert get_unscoped_file_storage().exists(lifecycle[-2])


def test_preview_symlink_boundary_preserves_target_and_handle(lifecycle, tmp_path):
    sessions, _, _, previews, _, _ = lifecycle
    outside = tmp_path / "outside-preview"
    outside.mkdir()
    target = outside / previews[0].name
    target.write_bytes(b"external content")
    directory = previews[0].parent
    previews[0].unlink()
    directory.rmdir()
    directory.symlink_to(outside, target_is_directory=True)
    assert collect(lifecycle).deleted == 0
    retained(lifecycle, ["durable", "local"])
    assert target.read_bytes() == b"external content"


def test_cleanup_claim_cannot_settle_on_durable_absence_alone(lifecycle):
    candidate, token = claim(lifecycle)
    with lifecycle[0]() as db:
        assert (
            store.settle_uploaded_file_compensation_no_commit(
                db,
                row_id=candidate.row_id,
                user_id=1,
                file_id=candidate.file_id,
                task_id=None,
                storage_key=candidate.storage_key,
                expected_updated_at=token,
                presence="absent",
            )
            is None
        )
        db.commit()
    retained(lifecycle, [])


def test_two_independent_workers_wait_for_inflight_delete_and_only_one_settles(
    lifecycle, monkeypatch
):
    from filelock import FileLock, Timeout

    candidate, token = claim(lifecycle)
    sessions, *_ = lifecycle
    entered, proceed, blocked = Event(), Event(), Event()
    original_delete = store.delete_uploaded_file_compensation_object
    original_acquire = FileLock.acquire
    calls = []

    def delete(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert proceed.wait(10)
        return original_delete(**kwargs)

    def acquire(lock, *args, **kwargs):
        try:
            return original_acquire(lock, *args, **{**kwargs, "timeout": 0})
        except Timeout:
            blocked.set()
            return original_acquire(lock, *args, **kwargs)

    monkeypatch.setattr(FileLock, "acquire", acquire)

    def run():
        return cleanup.run_uploaded_file_cleanup(
            session_factory=sessions,
            row_id=candidate.row_id,
            user_id=1,
            file_id=candidate.file_id,
            task_id=None,
            storage_key=candidate.storage_key,
            expected_updated_at=token,
            compensation_delete=delete,
            take_over=True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(run)
        try:
            assert entered.wait(10)
            loser = pool.submit(run)
            assert blocked.wait(10)
            with sessions() as db:
                assert db.query(UploadedFile).one().storage_status == "compensating"
        finally:
            proceed.set()
        assert winner.result(timeout=10) == "deleted"
        assert loser.result(timeout=10) == "stale"
    assert len(calls) == 1
    completed(lifecycle)


def test_stale_settlement_and_worker_cannot_touch_newer_claim(lifecycle):
    candidate, old = claim(lifecycle)
    sessions, _, _, _, key, file_id = lifecycle
    with file_cleanup_lock(file_id), sessions() as db:
        newer = store.take_over_uploaded_file_compensation_no_commit(
            db,
            row_id=candidate.row_id,
            user_id=1,
            file_id=file_id,
            task_id=None,
            storage_key=key,
            expected_updated_at=old,
        )
        assert newer != old
        db.commit()
    calls = []
    assert (
        cleanup.run_uploaded_file_cleanup(
            session_factory=sessions,
            row_id=candidate.row_id,
            user_id=1,
            file_id=file_id,
            task_id=None,
            storage_key=key,
            expected_updated_at=old,
            compensation_delete=lambda **kwargs: calls.append(kwargs) or "absent",
        )
        == "stale"
    )
    assert calls == []
    with sessions() as db:
        assert (
            store.settle_uploaded_file_compensation_no_commit(
                db,
                row_id=candidate.row_id,
                user_id=1,
                file_id=file_id,
                task_id=None,
                storage_key=key,
                expected_updated_at=old,
                presence="absent",
            )
            is None
        )
    assert recover(lifecycle).deleted == 1
    completed(lifecycle)


def test_managed_copy_loaded_before_claim_cannot_publish_after_cleanup(lifecycle):
    sessions, source, materialized, _, _, file_id = lifecycle
    with sessions() as db:
        ref = ManagedFileRef(db.query(UploadedFile).one())
        # Keep the caller's snapshot alive across cleanup on another connection.
        db.expunge_all()
    assert collect(lifecycle).deleted == 1
    with pytest.raises(FilePublicationUnavailable):
        ref.ensure_local()
    with pytest.raises(FilePublicationUnavailable):
        ref.materialize(allow_existing_local=False)
    assert not source.exists() and not materialized.exists()


def test_existing_copy_publication_blocks_claim_until_it_finishes(
    lifecycle, monkeypatch
):
    from filelock import FileLock, Timeout

    sessions, source, materialized, _, key, file_id = lifecycle
    source.unlink()
    with sessions() as db:
        row = db.query(UploadedFile).one()
        ref = ManagedFileRef(row)
        db.expunge_all()
    entered, proceed, blocked = Event(), Event(), Event()
    original = get_unscoped_file_storage().copy_to_path
    original_acquire = FileLock.acquire

    def copy_to_path(*args, **kwargs):
        entered.set()
        assert proceed.wait(10)
        return original(*args, **kwargs)

    def acquire(lock, *args, **kwargs):
        try:
            return original_acquire(lock, *args, **{**kwargs, "timeout": 0})
        except Timeout:
            blocked.set()
            return original_acquire(lock, *args, **kwargs)

    monkeypatch.setattr(get_unscoped_file_storage(), "copy_to_path", copy_to_path)
    monkeypatch.setattr(FileLock, "acquire", acquire)
    with ThreadPoolExecutor(max_workers=2) as pool:
        publication = pool.submit(ref.ensure_local)
        try:
            assert entered.wait(10)
            collection = pool.submit(collect, lifecycle)
            assert blocked.wait(10)
        finally:
            proceed.set()
        assert publication.result(timeout=10) == source
        assert collection.result(timeout=10).deleted == 1
    completed(lifecycle)


def test_retired_worker_cannot_delete_a_fresh_publication(lifecycle):
    sessions, source, _, _, key, file_id = lifecycle
    candidate, token = claim(lifecycle)
    assert recover(lifecycle).deleted == 1
    source.write_bytes(b"new published bytes")
    new_key = f"users/1/uploads/{file_id}/_versions/{uuid4()}/source.txt"
    staged = store.stage_uploaded_file_from_local_path(
        local_path=source,
        user_id=1,
        file_id=file_id,
        storage_key=new_key,
    )
    with sessions.begin() as db:
        store.UploadedFileStore(db).add_already_durable(staged.to_record())
    assert (
        cleanup.run_uploaded_file_cleanup(
            session_factory=sessions,
            row_id=candidate.row_id,
            user_id=1,
            file_id=file_id,
            task_id=None,
            storage_key=key,
            expected_updated_at=token,
            compensation_delete=store.delete_uploaded_file_compensation_object,
        )
        == "stale"
    )
    assert source.read_bytes() == b"new published bytes"
    assert get_unscoped_file_storage().exists(new_key)


def test_production_compensation_local_failure_preserves_original_contract(
    lifecycle, monkeypatch
):
    sessions, source, _, _, key, file_id = lifecycle
    with monkeypatch.context() as failing:
        failing.setattr(
            resources.os,
            "unlink",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unlink failed")),
        )
        with pytest.raises(store.DurableStorageOperationError):
            store.compensate_registered_uploads_sync(
                [store.RegisteredUploadCompensationClaim(1, file_id, None, key)]
            )
    retained(lifecycle, ["durable"])
    assert recover(lifecycle).deleted == 1
    completed(lifecycle)


@pytest.mark.parametrize(
    "phase", ["claim", "durable", "local", "previews", "quarantine"]
)
def test_process_exit_and_new_worker_converge(lifecycle, monkeypatch, phase):
    import subprocess
    import sys

    sessions, *_ = lifecycle
    engine = sessions.kw["bind"]
    if engine.dialect.name == "postgresql":
        with engine.connect() as connection:
            name = connection.execute(sa.text("SELECT current_database()")).scalar()
        url = sa.engine.make_url(os.environ["XAGENT_TEST_POSTGRES_URL"]).set(
            database=name
        )
    else:
        url = engine.url
    monkeypatch.setenv(
        "CLEANUP_TEST_DATABASE", url.render_as_string(hide_password=False)
    )
    script = """
import os, sys
from datetime import UTC, datetime
from xagent.web.models.database import configure_db, get_session_local
from xagent.web.services import uploaded_file_cleanup as cleanup
from xagent.web.services import uploaded_file_cleanup_resources as resources
from xagent.web.services.orphan_upload_gc import _claim_orphan, _orphan_candidates, cleanup_orphaned_taskless_uploads
configure_db(os.environ["CLEANUP_TEST_DATABASE"])
phase = sys.argv[1]
if phase == "claim":
    with get_session_local()() as db:
        candidate = _orphan_candidates(db, cutoff=datetime.now(UTC), limit=1, after=None)[0]
        _claim_orphan(db, candidate)
    os._exit(37)
if phase == "quarantine":
    original_unlink = resources.os.unlink
    def stop(path, *args, **kwargs):
        if ".cleanup-" in str(path):
            os._exit(37)
        return original_unlink(path, *args, **kwargs)
    resources.os.unlink = stop
else:
    original = cleanup._save_manifest
    def stop(*args, **kwargs):
        result = original(*args, **kwargs)
        if phase in args[-1]["done"]:
            os._exit(37)
        return result
    cleanup._save_manifest = stop
with get_session_local()() as db:
    cleanup_orphaned_taskless_uploads(db, older_than_seconds=1)
raise AssertionError("Did not interrupt the requested phase")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, phase],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 37, result.stderr
    expected = {
        "claim": [],
        "durable": ["durable"],
        "local": ["durable", "local"],
        "previews": ["durable", "local", "previews"],
        "quarantine": ["durable"],
    }[phase]
    manifest = retained(lifecycle, expected)
    if phase == "quarantine":
        resource = manifest["local"][0]
        assert (Path(resource["path"]).parent / resource["quarantine"]).exists()
    assert recover(lifecycle).deleted == 1
    completed(lifecycle)


def test_populated_upgrade_fresh_schema_retry_and_rollback_constraint(lifecycle):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    from tests.shared.postgres_disposable import load_migration_module

    sessions, *_ = lifecycle
    engine = sessions.kw["bind"]
    migration = load_migration_module(
        Path(
            "src/xagent/migrations/versions/20261005_uploaded_file_cleanup_manifest.py"
        )
    )
    fresh = sa.inspect(engine).get_columns("uploaded_files")
    with sessions.begin() as db:
        db.query(UploadedFile).one().cleanup_manifest = {}
    with sessions.begin() as db:
        db.query(UploadedFile).one().cleanup_manifest = None
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        assert connection.execute(
            sa.text("SELECT cleanup_manifest IS NULL FROM uploaded_files")
        ).scalar()
        migration.downgrade()
    assert "cleanup_manifest" not in {
        column["name"] for column in sa.inspect(engine).get_columns("uploaded_files")
    }
    with pytest.raises(OSError), engine.begin() as connection:
        if engine.dialect.name == "sqlite":
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            raise OSError("interrupted upgrade")
    assert "cleanup_manifest" not in {
        column["name"] for column in sa.inspect(engine).get_columns("uploaded_files")
    }
    for _ in range(2):
        with (
            engine.begin() as connection,
            Operations.context(MigrationContext.configure(connection)),
        ):
            migration.upgrade()
    upgraded = sa.inspect(engine).get_columns("uploaded_files")
    assert sorted(
        (c["name"], str(c["type"]), c["nullable"]) for c in upgraded
    ) == sorted([(c["name"], str(c["type"]), c["nullable"]) for c in fresh])
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.storage_status == "available" and row.checksum == DIGEST
    claim(lifecycle)
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        with pytest.raises(RuntimeError, match="Finish pending upload cleanup"):
            migration.downgrade()
    assert recover(lifecycle).deleted == 1
    with (
        engine.begin() as connection,
        Operations.context(MigrationContext.configure(connection)),
    ):
        migration.downgrade()
    with engine.connect() as connection:
        assert (
            connection.execute(
                sa.text("SELECT count(*) FROM uploaded_file_cleanup_fences")
            ).scalar()
            == 1
        )


def test_migration_refuses_offline_ddl():
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    from tests.shared.postgres_disposable import load_migration_module

    migration = load_migration_module(
        Path(
            "src/xagent/migrations/versions/20261005_uploaded_file_cleanup_manifest.py"
        )
    )
    with Operations.context(
        MigrationContext.configure(dialect_name="postgresql", opts={"as_sql": True})
    ):
        for operation in (migration.upgrade, migration.downgrade):
            with pytest.raises(RuntimeError, match="online migration"):
                operation()


def test_abandoned_managed_copy_and_converter_temporaries_are_owned(lifecycle):
    _, source, materialized, previews, _, file_id = lifecycle
    source_prefix = hashlib.sha256(file_id.encode()).hexdigest()[:24]
    temporary = [
        source.with_name(f".{source_prefix}.{source.name}.interrupted.tmp"),
        materialized.with_name(f".{materialized.name}.interrupted.tmp"),
        previews[0].with_name(f"{previews[0].name}.interrupted.tmp"),
    ]
    converter = previews[0].parent / f".{file_id}.preview-interrupted"
    converter.mkdir()
    (converter / "partial.pdf").write_bytes(b"partial conversion")
    for path in temporary:
        path.write_bytes(b"partial copy")
    unrelated = source.with_name(f".{source.name}.legacy-unknown.tmp")
    unrelated.write_bytes(b"unproven ownership")
    assert collect(lifecycle).deleted == 1
    completed(lifecycle)
    assert not any(path.exists() for path in temporary)
    assert not converter.exists()
    assert unrelated.read_bytes() == b"unproven ownership"


def test_preview_directory_partial_failure_can_be_retried(lifecycle, monkeypatch):
    _, _, _, previews, _, file_id = lifecycle
    converter = previews[0].parent / f".{file_id}.preview-interrupted"
    converter.mkdir()
    for name in ("first.pdf", "second.pdf"):
        (converter / name).write_bytes(b"partial")
    unlink = resources.os.unlink
    deleted = Event()

    def fail_after_first(path, *args, **kwargs):
        if str(path).endswith(".pdf") and str(path) in {"first.pdf", "second.pdf"}:
            if deleted.is_set():
                raise OSError("directory disposal interrupted")
            deleted.set()
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as failing:
        failing.setattr(resources.os, "unlink", fail_after_first)
        assert collect(lifecycle).deleted == 0
    retained(lifecycle, ["durable", "local"])
    assert deleted.is_set()
    assert recover(lifecycle).deleted == 1
    completed(lifecycle)
    assert not converter.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("protection", ["task", "target", "uncertain_delete"])
async def test_cancelled_production_upload_preserves_skipped_or_pending_claim(
    lifecycle, monkeypatch, protection
):
    import asyncio
    import io

    from fastapi.datastructures import UploadFile

    from xagent.web.api import files
    from xagent.web.models.kb_ingest_target import KBIngestTarget
    from xagent.web.models.task import Task
    from xagent.web.services.file_turn import bind_turn_files_no_commit

    sessions = lifecycle[0]
    registered, release = Event(), Event()
    identifiers = []
    original = files.register_local_uploads_sync

    def register_then_pause(registrations):
        result = original(registrations)
        identifiers.extend(item.file_id for item in result)
        with sessions.begin() as db:
            if protection == "task":
                task = Task(user_id=1, title="owns cancelled upload")
                db.add(task)
                db.flush()
                assert not bind_turn_files_no_commit(
                    file_ids=identifiers, task_id=int(task.id), owner_user_id=1, db=db
                )
            elif protection == "target":
                db.add(
                    KBIngestTarget(
                        user_id=1,
                        collection="protected",
                        target_path="cancelled.txt",
                        file_id=identifiers[0],
                        latest_file_sha256=DIGEST,
                    )
                )
        registered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(files, "register_local_uploads_sync", register_then_pause)
    if protection == "uncertain_delete":

        def fail_delete(*_args):
            raise OSError("delete unavailable")

        monkeypatch.setattr(type(get_unscoped_file_storage()), "delete", fail_delete)
    worker = asyncio.create_task(
        files.store_uploaded_files(
            upload_items=[
                UploadFile(filename="cancelled.txt", file=io.BytesIO(PAYLOAD))
            ],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=1,
            single_file_mode=True,
        )
    )
    assert await asyncio.to_thread(registered.wait, 10)
    worker.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await worker
    with sessions() as db:
        row = db.query(UploadedFile).filter_by(file_id=identifiers[0]).one()
        assert Path(row.storage_path).read_bytes() == PAYLOAD
        assert get_unscoped_file_storage().exists(row.storage_key)
        if protection == "uncertain_delete":
            assert row.storage_status == "compensating"
            assert row.cleanup_manifest["done"] == []
        else:
            assert row.storage_status == "available"
            assert row.cleanup_manifest is None


@pytest.mark.asyncio
async def test_request_finalizer_preserves_replacement_after_compensation(
    lifecycle, monkeypatch
):
    import io

    from fastapi.datastructures import UploadFile

    from xagent.web.api import files

    source = []
    original = files.compensate_registered_uploads_sync

    def fail_register(registrations):
        source.extend(item.local_path for item in registrations)
        raise RuntimeError("failed registration")

    def replace_after_compensation(claims):
        original(claims)
        source[0].rename(source[0].with_suffix(".retired"))
        source[0].write_bytes(b"new publication")

    monkeypatch.setattr(files, "register_local_uploads_sync", fail_register)
    monkeypatch.setattr(
        files, "compensate_registered_uploads_sync", replace_after_compensation
    )
    with pytest.raises(RuntimeError, match="failed registration"):
        await files.store_uploaded_files(
            upload_items=[
                UploadFile(filename="replacement.txt", file=io.BytesIO(PAYLOAD))
            ],
            task_type="general",
            task_id=None,
            folder=None,
            user_id=1,
            single_file_mode=True,
        )
    assert source[0].read_bytes() == b"new publication"


@pytest.mark.parametrize("first", ["bind", "claim"])
def test_independent_task_binder_and_claim_have_one_winner(lifecycle, first):
    from threading import local

    from xagent.web.models.task import Task
    from xagent.web.services.file_turn import bind_turn_files_no_commit

    sessions, source, _, _, _, file_id = lifecycle
    with sessions.begin() as db:
        task = Task(user_id=1, title="racing attachment")
        db.add(task)
        db.flush()
        task_id = int(task.id)
    with sessions() as db:
        candidate = _orphan_candidates(
            db, cutoff=datetime.now(UTC), limit=1, after=None
        )[0]
    entered, proceed, attempted = Event(), Event(), Event()
    worker = local()
    engine = sessions.kw["bind"]

    def before(_conn, _cursor, statement, *_args):
        if getattr(worker, "role", None) == "loser" and statement.startswith("UPDATE"):
            attempted.set()

    def after(_conn, _cursor, statement, *_args):
        if getattr(worker, "role", None) == "winner" and statement.startswith(
            "UPDATE uploaded_files"
        ):
            entered.set()
            assert proceed.wait(10)

    sa.event.listen(engine, "before_cursor_execute", before)
    sa.event.listen(engine, "after_cursor_execute", after)

    def execute(action, role):
        worker.role = role
        with sessions() as db:
            if action == "bind":
                missing = bind_turn_files_no_commit(
                    file_ids=[file_id], task_id=task_id, owner_user_id=1, db=db
                )
                db.commit()
                return missing
            return _claim_orphan(db, candidate)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            winner = pool.submit(execute, first, "winner")
            try:
                assert entered.wait(10)
                loser = pool.submit(
                    execute, "claim" if first == "bind" else "bind", "loser"
                )
                assert attempted.wait(10)
                assert not loser.done()
            finally:
                proceed.set()
            winning, losing = winner.result(timeout=10), loser.result(timeout=10)
        if first == "bind":
            assert winning == [] and losing is None
            assert source.read_bytes() == PAYLOAD
            with sessions() as db:
                row = db.query(UploadedFile).one()
                assert row.task_id == task_id and row.storage_status == "available"
        else:
            assert winning is not None and losing == [file_id]
            assert recover(lifecycle).deleted == 1
            completed(lifecycle)
    finally:
        sa.event.remove(engine, "before_cursor_execute", before)
        sa.event.remove(engine, "after_cursor_execute", after)


@pytest.mark.parametrize("target", ["source", "materialized", "preview"])
def test_file_replaced_by_directory_before_claim_is_preserved(lifecycle, target):
    _, source, materialized, previews, _, _ = lifecycle
    path = {"source": source, "materialized": materialized, "preview": previews[0]}[
        target
    ]
    path.unlink()
    path.mkdir()
    child = path / "unrelated.txt"
    child.write_bytes(b"replacement tree")
    assert collect(lifecycle).deleted == 0
    assert child.read_bytes() == b"replacement tree"
    retained(lifecycle, ["durable", "local"] if target == "preview" else [])


def test_missing_converter_child_does_not_settle_surviving_quarantine(
    lifecycle, monkeypatch
):
    _, _, _, previews, _, file_id = lifecycle
    directory = previews[0].parent / f".{file_id}.preview-interrupted"
    directory.mkdir()
    for name in ("first.pdf", "second.pdf"):
        (directory / name).write_bytes(b"partial preview")
    unlink = resources.os.unlink
    raced = Event()

    def disappear_before_unlink(path, *args, **kwargs):
        if str(path) in {"first.pdf", "second.pdf"} and not raced.is_set():
            raced.set()
            unlink(path, *args, **kwargs)
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as failing:
        failing.setattr(resources.os, "unlink", disappear_before_unlink)
        assert collect(lifecycle).deleted == 0
    assert raced.is_set()
    manifest = retained(lifecycle, ["durable", "local"])
    resource = next(r for r in manifest["previews"] if r["path"] == str(directory))
    quarantine = directory.parent / resource["quarantine"]
    assert len(list(quarantine.glob("*.pdf"))) == 1
    assert recover(lifecycle).deleted == 1
    completed(lifecycle)
    assert not quarantine.exists()


@pytest.mark.parametrize(
    "basename",
    ["source.txt", "r" * 206 + ".txt", "r" * 246 + ".txt", "还" * 80 + ".txt"],
)
def test_process_death_inside_real_restoration_cleans_both_temporaries(
    lifecycle, monkeypatch, basename
):
    import subprocess
    import sys

    sessions, source, materialized, previews, key, file_id = lifecycle
    if basename != source.name:
        renamed_source = source.with_name(basename)
        source.rename(renamed_source)
        renamed_cache = materialized.with_name(basename)
        materialized.rename(renamed_cache)
        with sessions.begin() as db:
            row = db.query(UploadedFile).one()
            row.storage_path, row.filename = str(renamed_source), basename
        source = renamed_source
        lifecycle = sessions, source, renamed_cache, previews, key, file_id
    engine = sessions.kw["bind"]
    if engine.dialect.name == "postgresql":
        with engine.connect() as connection:
            name = connection.execute(sa.text("SELECT current_database()")).scalar()
        url = sa.engine.make_url(os.environ["XAGENT_TEST_POSTGRES_URL"]).set(
            database=name
        )
    else:
        url = engine.url
    monkeypatch.setenv(
        "CLEANUP_TEST_DATABASE", url.render_as_string(hide_password=False)
    )
    source.unlink()
    script = """
import os
from xagent.core.file_storage import storage
from xagent.web.models.database import configure_db, get_session_local
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.managed_file_ref import ManagedFileRef
configure_db(os.environ["CLEANUP_TEST_DATABASE"])
def stop(source, destination, *args):
    destination.write(source.read(2))
    destination.flush()
    os._exit(37)
storage.shutil.copyfileobj = stop
with get_session_local()() as db:
    ManagedFileRef(db.query(UploadedFile).one()).ensure_local()
raise AssertionError("The actual restoration producer was not interrupted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=45
    )
    assert result.returncode == 37, result.stderr
    prefix = hashlib.sha256(file_id.encode()).hexdigest()[:24]
    copies = list(source.parent.iterdir())
    assert len(copies) == 2
    assert any(path.name.startswith(f".{prefix}.") for path in copies)
    assert any(path.name.startswith(f"..{prefix}.") for path in copies)
    assert collect(lifecycle).deleted == 1
    completed(lifecycle)
    assert not any(path.exists() for path in copies)


def test_migration_without_upload_table_defers_to_fresh_model_creation(lifecycle):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    from tests.shared.postgres_disposable import load_migration_module

    engine = lifecycle[0].kw["bind"]
    migration = load_migration_module(
        Path(
            "src/xagent/migrations/versions/20261005_uploaded_file_cleanup_manifest.py"
        )
    )
    with engine.begin() as connection:
        Base.metadata.drop_all(connection)
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            migration.upgrade()
            migration.downgrade()
        assert not sa.inspect(connection).has_table("uploaded_files")
        Base.metadata.create_all(connection)
        column = next(
            c
            for c in sa.inspect(connection).get_columns("uploaded_files")
            if c["name"] == "cleanup_manifest"
        )
        assert column["nullable"]
        assert isinstance(column["type"], sa.JSON)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["restore", "materialize", "svg", "pptx"])
async def test_publication_contention_preserves_caller_error_contract(
    lifecycle, monkeypatch, operation
):
    from types import SimpleNamespace

    from fastapi import HTTPException
    from filelock import FileLock, Timeout

    from xagent.web.api import files
    from xagent.web.services.managed_file_ref import DurableStorageOperationError

    sessions, source, materialized, _, key, file_id = lifecycle
    entered, release = Event(), Event()
    checked_out = [0]

    @sa.event.listens_for(sessions.kw["bind"], "checkout")
    def checkout(*_args):
        checked_out[0] += 1

    @sa.event.listens_for(sessions.kw["bind"], "checkin")  # codespell:ignore checkin
    def returned(*_args):
        checked_out[0] -= 1

    def slow_publication():
        with file_cleanup_lock(file_id):
            entered.set()
            assert release.wait(10)

    original_acquire = FileLock.acquire
    conversions = []

    async def unexpected_conversion(*args, **kwargs):
        conversions.append(args)
        raise AssertionError("Conversion must not start while publication is blocked")

    def bounded_acquire(lock, *args, **kwargs):
        if str(lock.lock_file).endswith(".cleanup.lock"):
            kwargs["timeout"] = 0.02
        return original_acquire(lock, *args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as pool:
        owner = pool.submit(slow_publication)
        assert entered.wait(5)
        monkeypatch.setattr(FileLock, "acquire", bounded_acquire)
        try:
            with sessions() as db:
                if operation in {"restore", "materialize"}:
                    source.unlink()
                    materialized.unlink()
                    ref = ManagedFileRef(db.query(UploadedFile).one())
                    with pytest.raises(DurableStorageOperationError) as fault:
                        ref.ensure_local() if operation == "restore" else ref.materialize()
                    assert fault.value.storage_key == key
                    assert isinstance(fault.value.__cause__, Timeout)
                elif operation == "svg":
                    with pytest.raises(HTTPException) as fault:
                        await files._inline_preview_response(
                            source,
                            filename="source.svg",
                            media_type="image/svg+xml",
                            file_id=file_id,
                        )
                    assert fault.value.status_code == 503
                    assert isinstance(
                        fault.value.__cause__, DurableStorageOperationError
                    )
                else:
                    pptx = source.with_suffix(".pptx")
                    row = db.query(UploadedFile).one()
                    row.filename = pptx.name
                    row.storage_path = str(pptx)
                    db.commit()
                    source.unlink()
                    monkeypatch.setattr(
                        files.asyncio, "create_subprocess_exec", unexpected_conversion
                    )
                    with pytest.raises(HTTPException) as fault:
                        await files.preview_pptx_as_pdf(
                            file_id, user=SimpleNamespace(id=1), db=db
                        )
                    assert fault.value.status_code == 503
                    assert isinstance(
                        fault.value.__cause__, DurableStorageOperationError
                    )
                    assert isinstance(fault.value.__cause__.__cause__, Timeout)
                    assert not conversions
                assert checked_out[0] == 0
        finally:
            release.set()
        owner.result(timeout=5)
    with sessions() as db:
        assert db.query(UploadedFile).one().storage_status == "available"
    if operation in {"restore", "materialize", "pptx"}:
        assert not source.exists()
    else:
        assert source.read_bytes() == PAYLOAD
