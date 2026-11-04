"""References and cleanup claims serialize through the production boundaries."""

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from threading import Event

import pytest
import sqlalchemy as sa
from filelock import FileLock
from sqlalchemy.orm import sessionmaker

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.providers.vector_store.lancedb import clear_connection_cache
from xagent.web.models.database import Base, configure_db, get_engine, get_session_local
from xagent.web.models.kb_ingest_target import KBIngestTarget
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.uploaded_file_cleanup_fence import UploadedFileCleanupFence
from xagent.web.models.user import User
from xagent.web.services.kb_ingest_targets import (
    admit_kb_ingest_target,
    release_kb_ingest_target_generation,
    tombstone_kb_ingest_target,
    tombstone_kb_ingest_targets_for_collection,
)
from xagent.web.services.kb_reference_protection import FileReferenceConflict
from xagent.web.services.orphan_upload_gc import (
    TASKLESS_SHARE_UPLOAD_SOURCE,
    cleanup_orphaned_taskless_uploads,
)


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def boundary(request, tmp_path, monkeypatch):
    from tests.shared.postgres_disposable import disposable_database_factory
    from xagent.web.models import database
    from xagent.web.services.kb_reference_protection import (
        install_file_reference_validator,
    )

    previous = database._SessionLocal, database._engine
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "lance"))
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized"))
    clear_connection_cache()
    stack = ExitStack()
    if request.param == "sqlite":
        configure_db(f"sqlite:///{tmp_path / 'metadata.db'}")
        engine = get_engine()
    else:
        make = stack.enter_context(disposable_database_factory("kb_reference"))
        engine = make("boundary")
        database._engine = engine
        database._SessionLocal = sessionmaker(bind=engine, autoflush=False)
        install_file_reference_validator()
    Base.metadata.create_all(engine)
    sessions = get_session_local()
    path = tmp_path / "uploads" / "user_1" / "source.txt"
    path.parent.mkdir(parents=True)
    path.write_text("retained source")
    with sessions.begin() as db:
        db.add_all(
            [
                User(id=1, username="owner", password_hash="unused"),
                User(id=2, username="other", password_hash="unused"),
            ]
        )
        db.flush()
        db.add(
            UploadedFile(
                user_id=1,
                file_id="source",
                filename=path.name,
                storage_path=str(path),
                storage_key="users/1/uploads/source/source.txt",
                checksum=hashlib.sha256(b"retained source").hexdigest(),
                storage_status="available",
                upload_source=TASKLESS_SHARE_UPLOAD_SOURCE,
                created_at=datetime.now(UTC) - timedelta(days=10),
            )
        )
    monkeypatch.setattr(
        "xagent.web.services.orphan_upload_gc.delete_uploaded_file_compensation_object",
        lambda **_kwargs: "absent",
    )
    yield sessions, LanceDBVectorIndexStore(), path
    clear_connection_cache()
    engine.dispose()
    database._SessionLocal, database._engine = previous
    stack.close()


def document(*, owner=1, file_id="source", doc_id="doc"):
    return {
        "collection": "kb",
        "doc_id": doc_id,
        "file_id": file_id,
        "source_path": "/source",
        "file_type": "txt",
        "content_hash": "hash",
        "uploaded_at": datetime.now(UTC),
        "title": None,
        "language": None,
        "user_id": owner,
    }


@pytest.mark.parametrize("owner", [1, 2])
def test_existing_document_prevents_cleanup_and_local_unlink(boundary, owner):
    sessions, store, path = boundary
    store.upsert_documents([document(owner=owner)])
    with sessions() as db:
        result = cleanup_orphaned_taskless_uploads(db, older_than_seconds=1)
        assert result.deleted == 0
        assert db.query(UploadedFile).one().storage_status == "available"
    assert path.read_text() == "retained source"


def claim(sessions):
    with sessions() as db:
        return cleanup_orphaned_taskless_uploads(db, older_than_seconds=1)


def handle(store):
    from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
        LanceDBCollectionHandle,
    )
    from xagent.core.tools.core.RAG_tools.kb.models import (
        KBAccessMode,
        KBBackendCapabilities,
        KBCollectionContext,
        KBStorageBackend,
        KBUserScope,
    )

    return LanceDBCollectionHandle(
        KBCollectionContext(
            collection="kb",
            user_scope=KBUserScope(1, False),
            access_mode=KBAccessMode.WRITE,
            allow_create=True,
            hide_missing=False,
            metadata_store=None,
            vector_index_store=store,
            ingestion_status_store=None,
            main_pointer_store=None,
            backend=KBStorageBackend.LANCEDB,
            capabilities=KBBackendCapabilities.lancedb(),
        )
    )


def write(store, mode):
    from xagent.core.tools.core.RAG_tools.core.schemas import DocumentRecordDetail
    from xagent.core.tools.core.RAG_tools.kb.models import KBDocumentRowsSnapshot

    row = document()
    if mode == "sync":
        store.upsert_documents([row])
    elif mode == "async":
        asyncio.run(store.upsert_documents_async([row]))
    elif mode == "restore":
        handle(store).restore_document(DocumentRecordDetail.from_row(row))
    else:
        handle(store).restore_document_rows(
            KBDocumentRowsSnapshot("kb", ("doc",), {"documents": [row]}),
            user_id=1,
            is_admin=False,
        )


@pytest.mark.parametrize("mode", ["sync", "async", "restore", "snapshot"])
def test_claimed_and_settled_identity_rejects_late_production_writes(boundary, mode):
    sessions, store, _ = boundary
    assert claim(sessions).deleted == 1
    with pytest.raises(FileReferenceConflict):
        write(store, mode)
    assert store.list_document_records_by_file_ids(["source"]) == []


def test_new_standalone_and_republished_uploads_are_accepted(boundary):
    from xagent.web.services.uploaded_file_store import UploadedFileStore

    sessions, store, path = boundary
    assert claim(sessions).deleted == 1
    store.upsert_documents([document(file_id="standalone", doc_id="standalone")])
    path.write_text("new publication")
    with sessions.begin() as db:
        UploadedFileStore(db).add_already_durable(
            UploadedFile(
                user_id=1,
                file_id="source",
                filename=path.name,
                storage_path=str(path),
                storage_status="available",
                storage_key="durable/new-generation",
                checksum="new",
            )
        )
    store.upsert_documents([document()])
    assert {
        row.file_id
        for row in store.list_document_records_by_file_ids(["source", "standalone"])
    } == {"source", "standalone"}


def test_standalone_without_sql_rejects_retired_id_but_accepts_new_source(
    boundary, monkeypatch
):
    from xagent.core.tools.core.RAG_tools.storage import file_reference
    from xagent.web.services.uploaded_file_store import UploadedFileStore

    sessions, store, path = boundary
    assert claim(sessions).deleted == 1
    monkeypatch.setattr(file_reference, "_validator", None)
    with pytest.raises(RuntimeError, match="Source file was claimed"):
        store.upsert_documents([document()])
    store.upsert_documents([document(file_id="independent", doc_id="independent")])
    path.write_text("new publication")
    with sessions.begin() as db:
        UploadedFileStore(db).add_already_durable(
            UploadedFile(
                user_id=1,
                file_id="source",
                filename=path.name,
                storage_path=str(path),
                storage_status="available",
                storage_key="durable/new-generation",
                checksum="new",
            )
        )
    store.upsert_documents([document()])
    assert len(store.list_document_records_by_file_ids(["source"])) == 1


def test_republication_rollback_keeps_standalone_fence(boundary, monkeypatch):
    from xagent.core.tools.core.RAG_tools.storage import file_reference
    from xagent.web.services.uploaded_file_store import UploadedFileStore

    sessions, store, path = boundary
    assert claim(sessions).deleted == 1
    with sessions() as db:
        UploadedFileStore(db).add_already_durable(
            UploadedFile(
                user_id=1,
                file_id="source",
                filename=path.name,
                storage_path=str(path),
                storage_status="available",
                storage_key="durable/new",
                checksum="new",
            )
        )
        db.rollback()
    monkeypatch.setattr(file_reference, "_validator", None)
    with pytest.raises(RuntimeError, match="Source file was claimed"):
        store.upsert_documents([document()])


def admit(sessions, *, generation="one", file_id="source", job_id=None):
    from xagent.web.services.background_jobs import create_background_job

    with sessions() as db:
        if job_id is None:
            job = create_background_job(
                db,
                user_id=2,
                job_type="kb.ingest.document",
                payload={"file_id": file_id},
            )
            job_id = str(job.id)
        admit_kb_ingest_target(
            db,
            user_id=2,
            collection="kb",
            target_path="canonical/source",
            file_id=file_id,
            generation_id=generation,
            job_id=job_id,
            file_sha256="hash",
        )
        return job_id


@pytest.mark.parametrize("release", ["release", "tombstone", "collection"])
def test_pending_target_protects_until_actual_release(boundary, release):
    sessions, _, path = boundary
    admit(sessions)
    assert claim(sessions).deleted == 0
    assert path.exists()
    with sessions() as db:
        if release == "release":
            assert release_kb_ingest_target_generation(
                db,
                user_id=2,
                collection="kb",
                target_path="canonical/source",
                generation_id="one",
            )
        elif release == "tombstone":
            tombstone_kb_ingest_target(
                db,
                user_id=2,
                collection="kb",
                target_path="canonical/source",
            )
        else:
            assert (
                tombstone_kb_ingest_targets_for_collection(
                    db, user_id=2, collection="kb"
                )
                == 1
            )
    assert claim(sessions).deleted == 1


def test_generation_replacement_and_stale_release_do_not_drop_protection(boundary):
    sessions, _, _ = boundary
    admit(sessions)
    admit(sessions, generation="two")
    with sessions() as db:
        assert not release_kb_ingest_target_generation(
            db,
            user_id=2,
            collection="kb",
            target_path="canonical/source",
            generation_id="one",
        )
    assert claim(sessions).deleted == 0
    admit(sessions, generation="three", file_id="new-source")
    assert claim(sessions).deleted == 1


def test_document_deletion_releases_protection(boundary):
    sessions, store, _ = boundary
    store.upsert_documents([document(owner=2)])
    assert claim(sessions).deleted == 0
    store.delete_document_record(
        collection_name="kb", doc_id="doc", user_id=2, is_admin=False
    )
    assert claim(sessions).deleted == 1


def test_reference_query_failure_prevents_claim_and_unlink(boundary, monkeypatch):
    sessions, _, path = boundary

    def unavailable(_ids):
        raise OSError("reference query unavailable")

    monkeypatch.setattr(
        "xagent.web.services.kb_file_service.find_referenced_file_ids", unavailable
    )
    assert claim(sessions).deleted == 0
    assert path.exists()
    with sessions() as db:
        assert db.query(UploadedFile).one().storage_status == "available"
        assert db.query(UploadedFileCleanupFence).count() == 0


@pytest.mark.parametrize("mode", ["sync", "async", "restore", "snapshot", "target"])
@pytest.mark.parametrize("first", ["reference", "claim"])
def test_independent_connections_race_with_both_winners(
    boundary, monkeypatch, mode, first
):
    from xagent.core.tools.core.RAG_tools.storage import file_reference

    sessions, store, path = boundary
    entered, proceed, blocked = Event(), Event(), Event()
    job_id = None
    if mode == "target":
        from xagent.web.services.background_jobs import create_background_job

        with sessions() as db:
            job_id = str(
                create_background_job(
                    db,
                    user_id=2,
                    job_type="kb.ingest.document",
                    payload={"file_id": "source"},
                ).id
            )

    original_acquire = FileLock._acquire

    def acquire(lock):
        original_acquire(lock)
        if lock._context.lock_file_fd is None:
            blocked.set()

    monkeypatch.setattr(FileLock, "_acquire", acquire)

    validator = file_reference._validator

    def pause_reference(ids):
        validator(ids)
        if first == "reference":
            entered.set()
            assert proceed.wait(DB_PROGRESS_TIMEOUT)

    monkeypatch.setattr(file_reference, "_validator", pause_reference)

    @sa.event.listens_for(sessions.kw["bind"], "after_cursor_execute")
    def pause_claim(_conn, _cursor, statement, _parameters, _context, _many):
        if (
            first == "claim"
            and statement.startswith("UPDATE uploaded_files SET storage_status")
        ) or (
            first == "reference"
            and mode == "target"
            and statement.startswith("INSERT INTO kb_ingest_targets")
        ):
            entered.set()
            assert proceed.wait(DB_PROGRESS_TIMEOUT)

    def establish():
        if mode == "target":
            # Admission uses the caller's independent connection and checks the
            # same state as document publication, without the global validator.
            admit(sessions, job_id=job_id)
        else:
            write(store, mode)

    actions = {"reference": establish, "claim": lambda: claim(sessions)}

    def winning():
        return actions[first]()

    with ThreadPoolExecutor(max_workers=2) as executor:
        winner = executor.submit(winning)
        try:
            assert entered.wait(DB_PROGRESS_TIMEOUT)
            loser = executor.submit(
                actions["claim" if first == "reference" else "reference"]
            )
            assert blocked.wait(DB_PROGRESS_TIMEOUT), (
                "Contender did not actually wait on the winner's lock"
            )
        finally:
            proceed.set()
        result = winner.result(timeout=8)
        if first == "claim":
            assert result.deleted == 1
            with pytest.raises(FileReferenceConflict):
                loser.result(timeout=8)
            assert store.list_document_records_by_file_ids(["source"]) == []
        else:
            assert loser.result(timeout=8).deleted == 0
            assert path.exists()
    sa.event.remove(sessions.kw["bind"], "after_cursor_execute", pause_claim)


def test_reference_write_failure_releases_lock_and_does_not_pin(boundary, monkeypatch):
    sessions, store, _ = boundary
    original = store._get_connection

    def unavailable():
        raise OSError("publication failed")

    monkeypatch.setattr(store, "_get_connection", unavailable)
    with pytest.raises(OSError):
        store.upsert_documents([document()])
    monkeypatch.setattr(store, "_get_connection", original)
    assert claim(sessions).deleted == 1


def test_claim_commit_failure_rolls_back_fence_and_keeps_reference_possible(
    boundary, monkeypatch
):
    from xagent.web.services.orphan_upload_gc import _claim_orphan, _orphan_candidates

    sessions, store, _ = boundary
    with sessions() as db:
        candidate = _orphan_candidates(
            db, cutoff=datetime.now(UTC), limit=1, after=None
        )[0]

        def fail():
            raise OSError("commit failed")

        monkeypatch.setattr(db, "commit", fail)
        with pytest.raises(OSError):
            _claim_orphan(db, candidate)
        db.rollback()
    store.upsert_documents([document()])
    with sessions() as db:
        assert db.query(UploadedFile).one().storage_status == "available"
        assert db.query(UploadedFileCleanupFence).count() == 0


def test_async_cancellation_waits_for_native_publication(boundary, monkeypatch):
    sessions, store, _ = boundary
    original = store._get_async_connection

    async def scenario():
        started, proceed = asyncio.Event(), asyncio.Event()

        class Connection:
            async def open_table(self, name):
                table = await (await original()).open_table(name)

                class Table:
                    def merge_insert(self, keys):
                        builder = table.merge_insert(keys)

                        class Builder:
                            def when_matched_update_all(self):
                                builder.when_matched_update_all()
                                return self

                            def when_not_matched_insert_all(self):
                                builder.when_not_matched_insert_all()
                                return self

                            async def execute(self, rows):
                                started.set()
                                await proceed.wait()
                                await builder.execute(rows)

                        return Builder()

                    def close(self):
                        table.close()

                return Table()

        async def connection():
            return Connection()

        monkeypatch.setattr(store, "_get_async_connection", connection)
        publication = asyncio.create_task(store.upsert_documents_async([document()]))
        await asyncio.wait_for(started.wait(), DB_PROGRESS_TIMEOUT)
        publication.cancel()
        await asyncio.sleep(0)
        cleanup = asyncio.create_task(asyncio.to_thread(claim, sessions))
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await publication
        assert (await cleanup).deleted == 0
        assert len(store.list_document_records_by_file_ids(["source"])) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["before", "after", "claim_before_commit"])
def test_process_exit_releases_lock_without_losing_actual_reference(
    boundary, stage, monkeypatch
):
    import os
    import subprocess
    import sys

    from sqlalchemy.engine import make_url

    sessions, store, _ = boundary
    engine = sessions.kw["bind"]
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            name = conn.execute(sa.text("SELECT current_database()")).scalar()
        url = make_url(os.environ["XAGENT_TEST_POSTGRES_URL"]).set(database=name)
    else:
        url = engine.url
    monkeypatch.setenv(
        "REFERENCE_TEST_DATABASE", url.render_as_string(hide_password=False)
    )
    script = """
import os, sys
from datetime import UTC, datetime
from xagent.web.models.database import configure_db
from xagent.core.tools.core.RAG_tools.storage import file_reference
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import LanceDBVectorIndexStore
configure_db(os.environ["REFERENCE_TEST_DATABASE"])
if sys.argv[1] == "claim_before_commit":
    from xagent.web.models.database import get_session_local
    from xagent.web.services.orphan_upload_gc import _claim_orphan, _orphan_candidates
    with get_session_local()() as db:
        candidate = _orphan_candidates(db, cutoff=datetime.now(UTC), limit=1, after=None)[0]
        db.commit = lambda: os._exit(37)
        _claim_orphan(db, candidate)
store = LanceDBVectorIndexStore()
if sys.argv[1] == "before":
    original = file_reference._validator
    def exit_before(ids):
        original(ids)
        os._exit(37)
    file_reference._validator = exit_before
else:
    store.invalidate_table_cache = lambda *_args: os._exit(37)
store.upsert_documents([{
    "collection": "kb", "doc_id": "doc", "file_id": "source",
    "source_path": "/source", "file_type": "txt", "content_hash": "hash",
    "uploaded_at": datetime.now(UTC), "title": None, "language": None, "user_id": 1,
}])
"""
    process = subprocess.run(
        [sys.executable, "-c", script, stage], timeout=45, capture_output=True
    )
    assert process.returncode == 37, process.stderr.decode()
    if stage == "claim_before_commit":
        store.upsert_documents([document()])
        with sessions() as db:
            assert db.query(UploadedFileCleanupFence).count() == 0
    assert claim(sessions).deleted == (1 if stage == "before" else 0)


def test_sql_connections_are_returned_before_lancedb_writes(boundary, monkeypatch):
    sessions, store, _ = boundary
    active = set()
    engine = sessions.kw["bind"]

    def checkout(connection, _record, _proxy):
        active.add(id(connection))

    def returned(connection, _record):
        active.discard(id(connection))

    sa.event.listen(engine, "checkout", checkout)
    sa.event.listen(engine, "checkin", returned)  # codespell:ignore checkin
    original = store._get_connection

    def connection():
        assert not active, "SQL connection held during LanceDB publication"
        return original()

    monkeypatch.setattr(store, "_get_connection", connection)
    try:
        write(store, "sync")
        write(store, "async")
        write(store, "restore")
        write(store, "snapshot")
    finally:
        sa.event.remove(engine, "checkout", checkout)
        sa.event.remove(engine, "checkin", returned)  # codespell:ignore checkin


def test_released_reference_preserves_detachment_window(boundary):
    sessions, store, path = boundary
    detached = datetime.now(UTC) - timedelta(hours=1)
    with sessions.begin() as db:
        row = db.query(UploadedFile).one()
        row.detached_reason = "task_deleted"
        row.detached_at = detached
    store.upsert_documents([document()])
    store.delete_document_record(
        collection_name="kb", doc_id="doc", user_id=1, is_admin=False
    )
    assert claim(sessions).scanned == 0
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.detached_at.replace(tzinfo=UTC) == detached
    assert path.exists()


def test_registered_upload_compensation_also_protects_documents_and_targets(boundary):
    from xagent.web.services.uploaded_file_store import (
        RegisteredUploadCompensationClaim,
        compensate_registered_uploads_sync,
    )

    sessions, store, path = boundary
    store.upsert_documents([document(owner=2)])
    compensation = RegisteredUploadCompensationClaim(
        user_id=1,
        file_id="source",
        expected_task_id=None,
        expected_storage_key="users/1/uploads/source/source.txt",
    )
    compensate_registered_uploads_sync([compensation])
    store.delete_document_record(
        collection_name="kb", doc_id="doc", user_id=2, is_admin=False
    )
    admit(sessions)
    compensate_registered_uploads_sync([compensation])
    with sessions() as db:
        assert db.query(UploadedFile).one().storage_status == "available"
    assert path.exists()


def test_schema_upgrade_preserves_data_and_matches_fresh_install(boundary):
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    from tests.shared.postgres_disposable import load_migration_module

    sessions, _, _ = boundary
    engine = sessions.kw["bind"]
    migration = load_migration_module(
        Path("src/xagent/migrations/versions/20261005_uploaded_file_cleanup_fences.py")
    )
    fresh = sa.inspect(engine).get_columns("uploaded_file_cleanup_fences")
    UploadedFileCleanupFence.__table__.drop(engine)
    for _ in range(2):
        with engine.begin() as conn:
            context = MigrationContext.configure(conn)
            with Operations.context(context):
                migration.upgrade()
    upgraded = sa.inspect(engine).get_columns("uploaded_file_cleanup_fences")
    assert [(c["name"], str(c["type"]), c["nullable"]) for c in upgraded] == [
        (c["name"], str(c["type"]), c["nullable"]) for c in fresh
    ]
    with sessions() as db:
        assert db.query(UploadedFile).one().file_id == "source"
    assert claim(sessions).deleted == 1
    with engine.begin() as conn, Operations.context(MigrationContext.configure(conn)):
        migration.upgrade()
        with pytest.raises(RuntimeError, match="Retain cleanup fences"):
            migration.downgrade()
    with sessions() as db:
        assert db.query(UploadedFileCleanupFence).one().file_id == "source"


def test_schema_failure_retry_leaves_other_tables_intact(boundary):
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    from tests.shared.postgres_disposable import load_migration_module

    sessions, _, _ = boundary
    engine = sessions.kw["bind"]
    migration = load_migration_module(
        Path("src/xagent/migrations/versions/20261005_uploaded_file_cleanup_fences.py")
    )
    UploadedFileCleanupFence.__table__.drop(engine)
    with pytest.raises(OSError), engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.exec_driver_sql("BEGIN IMMEDIATE")
        with Operations.context(MigrationContext.configure(conn)):
            migration.upgrade()
            raise OSError("interrupted migration")
    assert not sa.inspect(engine).has_table("uploaded_file_cleanup_fences")
    with engine.begin() as conn, Operations.context(MigrationContext.configure(conn)):
        migration.upgrade()
    with sessions() as db:
        assert db.query(UploadedFile).one().file_id == "source"


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("release_failure", [False, True])
def test_real_staged_ingest_releases_final_target_and_preserves_actual_documents(
    boundary,
    tmp_path,
    monkeypatch,
    failure,
    release_failure,
):
    from xagent.core.file_storage.factory import get_unscoped_file_storage
    from xagent.core.model.model import EmbeddingModelConfig
    from xagent.core.storage.manager import initialize_storage_manager
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig
    from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion
    from xagent.web.jobs.exceptions import BackgroundJobHandlerError
    from xagent.web.jobs.kb_tasks import handle_kb_ingest_document
    from xagent.web.models import database
    from xagent.web.services import kb_ingest_targets
    from xagent.web.services.background_jobs import create_background_job

    sessions, store, path = boundary
    foreign_engine = sa.create_engine(f"sqlite:///{tmp_path / 'foreign.db'}")
    Base.metadata.create_all(foreign_engine)
    foreign_sessions = sessionmaker(bind=foreign_engine)
    with foreign_sessions.begin() as foreign:
        foreign.add(User(id=1, username="foreign", password_hash="unused"))
        foreign.flush()
        foreign.add(
            UploadedFile(
                user_id=1,
                file_id="source",
                filename="foreign.txt",
                storage_path="/foreign",
                storage_status="compensating",
            )
        )
    release = kb_ingest_targets.release_kb_ingest_target_generation

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("target release unavailable")

    if release_failure:
        monkeypatch.setattr(
            kb_ingest_targets, "release_kb_ingest_target_generation", unavailable
        )
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "durable").as_uri())
    get_unscoped_file_storage.cache_clear()

    initialize_storage_manager(str(tmp_path / "workspace"), str(tmp_path / "uploads"))

    class Embeddings:
        abilities = ["embedding"]

        def get_dimension(self):
            return 2

        def encode(self, text, **_kwargs):
            if failure:
                raise RuntimeError("embedding provider failed")
            if isinstance(text, str):
                return [float(len(text)), 0.0]
            return [[float(len(item)), 0.0] for item in text]

    config = EmbeddingModelConfig(
        id="test", model_name="test", model_provider="test", dimension=2
    )
    monkeypatch.setattr(
        document_ingestion,
        "_resolve_embedding_adapter",
        lambda _cfg: (config, Embeddings()),
    )
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.management.collection_manager.resolve_embedding_adapter",
        lambda *_args, **_kwargs: (config, Embeddings()),
    )
    staged = tmp_path / "stage.txt"
    staged.write_text("The real staged ingestion source.")
    with sessions() as db:
        job = create_background_job(
            db,
            user_id=1,
            job_type="kb.ingest.document",
            payload={
                "collection": "kb",
                "source_path": str(staged),
                "target_path": str(path),
                "file_id": "source",
                "generation_id": "one",
                "user_id": 1,
                "is_admin": False,
                "filename": path.name,
                "mime_type": "text/plain",
                "file_size": staged.stat().st_size,
                "ingestion_config": IngestionConfig(
                    embedding_model_id="test"
                ).model_dump(mode="json"),
                "collection_existed_before": True,
                "document_existed_before": False,
            },
        )
        job_id = str(job.id)
        admit_kb_ingest_target(
            db,
            user_id=1,
            collection="kb",
            target_path=str(path),
            file_id="source",
            generation_id="one",
            job_id=job_id,
            file_sha256="hash",
        )
        job = db.get(type(job), job_id)
        job.attempts = job.max_attempts
        db.commit()
        with monkeypatch.context() as global_override:
            global_override.setattr(database, "_SessionLocal", foreign_sessions)
            if failure:
                with pytest.raises(
                    BackgroundJobHandlerError, match="embedding provider failed"
                ):
                    handle_kb_ingest_document(db, job)
            else:
                assert handle_kb_ingest_document(db, job)["status"] == "success"
        assert (db.query(KBIngestTarget).one().deleted_at is None) is release_failure
    assert bool(store.list_document_records_by_file_ids(["source"])) is not failure
    if release_failure:
        assert claim(sessions).deleted == 0
        with sessions() as db:
            assert release(
                db,
                user_id=1,
                collection="kb",
                target_path=str(path),
                generation_id="one",
            )
    assert claim(sessions).deleted == (1 if failure else 0)
    foreign_engine.dispose()
    get_unscoped_file_storage.cache_clear()


@pytest.fixture
def submit_ingest(boundary, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from xagent.web.api import kb
    from xagent.web.auth_dependencies import get_current_user
    from xagent.web.models.database import get_db

    sessions, _store, path = boundary
    app = FastAPI()
    app.include_router(kb.kb_router)

    def db_dependency():
        with sessions() as db:
            yield db

    app.dependency_overrides[get_db] = db_dependency
    app.dependency_overrides[get_current_user] = lambda: User(
        id=1, username="owner", is_admin=False
    )

    async def available(*_args, **_kwargs):
        return None

    async def enqueue(_db, job):
        return job

    monkeypatch.setattr(kb, "_ensure_collection_access", available)
    monkeypatch.setattr(kb, "_ensure_background_job_queue_available_async", available)
    monkeypatch.setattr(kb, "_enqueue_background_job_or_503_async", enqueue)
    monkeypatch.setattr(kb, "get_upload_path", lambda *_args, **_kwargs: str(path))
    monkeypatch.setattr(
        kb,
        "_build_background_ingest_staging_path",
        lambda **_kwargs: tmp_path / "staged.txt",
    )
    client = TestClient(app, raise_server_exceptions=False)

    def submit():
        return client.post(
            "/api/kb/ingest/jobs",
            files={"file": ("source.txt", b"fresh upload", "text/plain")},
            data={"collection": "kb"},
        )

    return submit


@pytest.mark.parametrize("retired", [False, True])
def test_real_upload_admission_preserves_unretired_id_and_renews_retired_id(
    boundary, submit_ingest, retired
):
    from xagent.web.api.kb import _background_ingest_file_id
    from xagent.web.models.background_job import BackgroundJob

    sessions, store, path = boundary
    identity = _background_ingest_file_id(user_id=1, storage_path=path)
    with sessions.begin() as db:
        db.query(UploadedFile).update({UploadedFile.file_id: identity})
        if not retired:
            db.query(UploadedFile).delete()
    if retired:
        assert claim(sessions).deleted == 1
    response = submit_ingest()
    assert response.status_code == 202, response.text
    with sessions() as db:
        job = db.get(BackgroundJob, response.json()["id"])
        chosen = job.payload["file_id"]
        assert (chosen != identity) is retired
        assert db.query(KBIngestTarget).one().file_id == chosen
    store.upsert_documents([document(file_id=chosen)])
    assert store.list_document_records_by_file_ids([chosen])


def test_admission_lock_failure_marks_job_failed_and_allows_fresh_retry(
    boundary, submit_ingest, monkeypatch, tmp_path
):
    from filelock import Timeout

    from xagent.web.models.background_job import BackgroundJob
    from xagent.web.services import kb_reference_protection

    sessions, _store, _path = boundary
    original = kb_reference_protection.file_reference_lock

    def unavailable(_ids):
        raise Timeout(str(tmp_path / "private-reference.lock"))

    monkeypatch.setattr(kb_reference_protection, "file_reference_lock", unavailable)
    response = submit_ingest()
    assert response.status_code == 503
    assert response.json() == {"detail": "KB source is busy; retry the upload"}
    assert response.headers["retry-after"] == "15"
    assert str(tmp_path) not in response.text
    with sessions() as db:
        failed = db.query(BackgroundJob).one()
        assert failed.status == "failed"
        assert failed.error_message == "KB source is busy; retry the upload"
        assert db.query(KBIngestTarget).count() == 0
    assert not (tmp_path / "staged.txt").exists()
    monkeypatch.setattr(kb_reference_protection, "file_reference_lock", original)
    response = submit_ingest()
    assert response.status_code == 202, response.text
    assert response.json()["id"] != failed.id


def test_retryable_worker_failure_keeps_generation_and_staged_source(
    boundary, tmp_path, monkeypatch
):
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig
    from xagent.web.jobs import kb_tasks
    from xagent.web.jobs.exceptions import BackgroundJobHandlerError
    from xagent.web.services.background_jobs import create_background_job

    sessions, store, path = boundary
    staged = tmp_path / "retry.txt"
    staged.write_text("retryable source")
    error = BackgroundJobHandlerError("temporary ingest failure", retryable=True)

    def temporarily_unavailable(**_kwargs):
        raise error

    monkeypatch.setattr(kb_tasks, "run_document_ingestion", temporarily_unavailable)
    with sessions() as db:
        job = create_background_job(
            db,
            user_id=1,
            job_type="kb.ingest.document",
            payload={
                "user_id": 2,
                "collection": "kb",
                "file_id": "source",
                "target_path": "canonical/source",
                "source_path": str(staged),
                "generation_id": "one",
                "collection_existed_before": True,
                "ingestion_config": IngestionConfig().model_dump(mode="json"),
            },
        )
        job_id = str(job.id)
    admit(sessions, job_id=job_id)
    with sessions() as db:
        job = db.get(type(job), job_id)
        job.attempts = 1
        db.commit()
        with pytest.raises(BackgroundJobHandlerError) as caught:
            kb_tasks.handle_kb_ingest_document(db, job)
        assert caught.value is error
        assert db.query(KBIngestTarget).one().deleted_at is None
    assert staged.read_text() == "retryable source"
    assert not store.list_document_records_by_file_ids(["source"])
    assert claim(sessions).deleted == 0
    assert path.exists()


def test_compensating_canonical_upload_rejects_admission_with_safe_conflict(
    boundary, submit_ingest, tmp_path
):
    from xagent.web.models.background_job import BackgroundJob

    sessions, store, _path = boundary
    with sessions.begin() as db:
        db.query(UploadedFile).update({UploadedFile.storage_status: "compensating"})
    response = submit_ingest()
    assert response.status_code == 409
    message = "KB source is unavailable or the ingest was superseded"
    assert response.json() == {"detail": message}
    with sessions() as db:
        failed = db.query(BackgroundJob).one()
        assert failed.status == "failed"
        assert failed.error_message == message
        assert db.query(KBIngestTarget).count() == 0
        assert db.query(UploadedFile).one().file_id == "source"
    assert not store.list_document_records_by_file_ids(["source"])
    assert not (tmp_path / "staged.txt").exists()


def test_upload_admission_wait_keeps_request_event_loop_responsive(
    boundary, submit_ingest, monkeypatch
):
    from contextlib import contextmanager

    from xagent.web.api import kb
    from xagent.web.services import kb_reference_protection

    tick = Event()
    request_loops = []
    original = kb_reference_protection.file_reference_lock

    async def access(*_args, **_kwargs):
        request_loops.append(asyncio.get_running_loop())

    @contextmanager
    def delayed_lock(ids):
        request_loops[0].call_soon_threadsafe(tick.set)
        assert tick.wait(3), "Admission blocked the request event loop"
        with original(ids):
            yield

    monkeypatch.setattr(kb, "_ensure_collection_access", access)
    monkeypatch.setattr(kb_reference_protection, "file_reference_lock", delayed_lock)
    response = submit_ingest()
    assert response.status_code == 202, response.text
    assert tick.is_set()
