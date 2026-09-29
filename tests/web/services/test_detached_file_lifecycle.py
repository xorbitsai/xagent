"""Prospective attachment retention at the task and file service boundaries."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.file_turn import bind_turn_files_no_commit
from xagent.web.services.task_deletion import purge_task_rows


@pytest.fixture
def sessions(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'detach.db'}")

    @sa.event.listens_for(engine, "connect")
    def enable_fks(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with factory() as db:
        owner = User(username="owner", password_hash="unused")
        db.add(owner)
        db.flush()
        db.add_all(
            Task(id=i, user_id=owner.id, title=str(i), status=TaskStatus.COMPLETED)
            for i in (1, 2)
        )
        db.flush()
        db.add_all(
            UploadedFile(
                file_id=name,
                user_id=owner.id,
                task_id=task_id,
                filename=f"{name}.txt",
                storage_path=str(tmp_path / f"{name}.txt"),
                storage_status="legacy",
                created_at=datetime.now(UTC) - timedelta(days=365),
            )
            for name, task_id in (("attached", 1), ("draft", None), ("other", 2))
        )
        db.commit()
    yield factory
    engine.dispose()


@pytest.mark.parametrize("reason", ["task_deleted", "task_create_failed"])
def test_delete_marks_only_its_attachments_and_rollback_preserves_binding(
    sessions, reason
):
    with sessions() as db:
        before = datetime.now(UTC).replace(tzinfo=None)
        assert purge_task_rows(db, task_id=1, detached_reason=reason)
        db.flush()
        row = db.query(UploadedFile).filter_by(file_id="attached").one()
        assert row.task_id is None
        assert row.detached_reason == reason
        assert row.detached_at >= before
        assert row.detached_at > row.created_at
        for name in ("draft", "other"):
            untouched = db.query(UploadedFile).filter_by(file_id=name).one()
            assert untouched.detached_at is None
            assert untouched.detached_reason is None
        db.rollback()
    with sessions() as db:
        row = db.query(UploadedFile).filter_by(file_id="attached").one()
        assert row.task_id == 1
        assert row.detached_at is None
        assert db.get(Task, 1) is not None


def test_explicit_rebind_clears_provenance_and_next_delete_restarts_window(sessions):
    with sessions() as db:
        purge_task_rows(db, task_id=1, detached_reason="task_deleted")
        db.commit()
        row = db.query(UploadedFile).filter_by(file_id="attached").one()
        first_detach = row.detached_at
        assert (
            bind_turn_files_no_commit(
                file_ids=["attached"], task_id=2, owner_user_id=1, db=db
            )
            == []
        )
        db.commit()
        db.refresh(row)
        assert row.task_id == 2
        assert row.detached_reason is None
        assert row.detached_at is None
        purge_task_rows(db, task_id=2, detached_reason="task_deleted")
        db.commit()
        db.refresh(row)
        assert row.detached_at > first_detach


@pytest.mark.parametrize("target,owner", [(999, 1), (2, 999)])
def test_missing_or_foreign_task_cannot_claim_draft(sessions, target, owner):
    with sessions() as db:
        assert bind_turn_files_no_commit(
            file_ids=["draft"], task_id=target, owner_user_id=owner, db=db
        ) == ["draft"]
        assert db.query(UploadedFile).filter_by(file_id="draft").one().task_id is None


def test_public_access_and_automatic_repair_reject_detached_file(sessions):
    from fastapi import HTTPException

    from xagent.web.api.files import _validate_public_task_file_access
    from xagent.web.services.file_reference_output_service import (
        load_assistant_file_reference_records,
    )

    with sessions() as db:
        purge_task_rows(db, task_id=1, detached_reason="task_deleted")
        db.commit()
        row = db.query(UploadedFile).filter_by(file_id="attached").one()
        with pytest.raises(HTTPException) as exc:
            _validate_public_task_file_access(db, row, None)
        assert exc.value.status_code == 403
        assert "attached" not in {
            str(record.file_id)
            for record in load_assistant_file_reference_records(
                db, user_id=1, task_id=2
            )
        }


def test_taskless_gc_preserves_detached_share_upload(sessions, tmp_path):
    from xagent.web.services.orphan_upload_gc import cleanup_orphaned_taskless_uploads

    path = tmp_path / "attached.txt"
    path.write_text("must survive")
    with sessions() as db:
        row = db.query(UploadedFile).filter_by(file_id="attached").one()
        row.upload_source = "taskless_share_upload"
        row.storage_status = "available"
        row.storage_key = "users/1/uploads/attached/attached.txt"
        db.commit()
        purge_task_rows(db, task_id=1, detached_reason="task_deleted")
        db.commit()
        result = cleanup_orphaned_taskless_uploads(db, older_than_seconds=48 * 3600)
        assert result.scanned == 0
        assert path.read_text() == "must survive"


def test_metadata_restore_preserves_detachment_and_rejects_an_old_generation(sessions):
    from dataclasses import replace

    from xagent.web.services.uploaded_file_store import (
        UploadedFileStore,
        UploadedFileVersionConflict,
        snapshot_uploaded_file_version,
    )

    with sessions() as db:
        purge_task_rows(db, task_id=1, detached_reason="task_deleted")
        db.commit()
        row = db.query(UploadedFile).filter_by(file_id="attached").one()
        snapshot = snapshot_uploaded_file_version(row)
        restored = (
            UploadedFileStore(db)
            .restore_metadata_version_no_commit(
                expected=snapshot, replacement=replace(snapshot, filename="renamed.txt")
            )
            .snapshot
        )
        db.commit()
        assert restored.detached_at == snapshot.detached_at
        assert restored.detached_reason == "task_deleted"
        assert (
            bind_turn_files_no_commit(
                file_ids=["attached"], task_id=2, owner_user_id=1, db=db
            )
            == []
        )
        purge_task_rows(db, task_id=2, detached_reason="task_deleted")
        db.commit()
        with pytest.raises(UploadedFileVersionConflict):
            UploadedFileStore(db).restore_metadata_version_no_commit(
                expected=restored, replacement=restored
            )


@pytest.mark.parametrize("uploads_only", [False, True])
def test_file_list_and_public_routes_hide_detached_but_owner_can_rebind(
    sessions, monkeypatch, uploads_only
):
    from uuid import uuid4

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from xagent.web.api import files

    app = FastAPI()
    app.include_router(files.file_router)
    file_id = str(uuid4())
    with sessions() as db:
        row = db.query(UploadedFile).filter_by(file_id="attached").one()
        row.file_id = file_id
        db.commit()
        purge_task_rows(db, task_id=1, detached_reason="task_deleted")
        db.commit()
        owner = db.get(User, 1)
        app.dependency_overrides[files.get_current_user] = lambda: owner
        app.dependency_overrides[files.get_db] = lambda: db
        monkeypatch.setattr(files, "aggregate_uploaded_file_statuses", lambda **_: {})
        with TestClient(app) as client:
            response = client.get(
                "/api/files/list", params={"uploads_only": uploads_only}
            )
            assert response.status_code == 200
            ids = {item["file_id"] for item in response.json()["files"]}
            assert "draft" in ids
            assert file_id not in ids
            for route in ("download", "preview"):
                assert (
                    client.get(f"/api/files/public/{route}/{file_id}").status_code
                    == 403
                )
            assert (
                bind_turn_files_no_commit(
                    file_ids=[file_id], task_id=2, owner_user_id=1, db=db
                )
                == []
            )
            db.commit()
            response = client.get("/api/files/list", params={"task_id": 2})
            assert file_id in {item["file_id"] for item in response.json()["files"]}


def test_failed_create_compensation_marks_only_committed_attachments(sessions):
    from xagent.web.api.chat import _compensate_failed_task_extension_create

    with sessions() as db:
        # A never-committed binding must be rolled back before compensating.
        db.query(UploadedFile).filter_by(file_id="draft").update({"task_id": 1})
        _compensate_failed_task_extension_create(db, task_id=1)
        assert db.get(Task, 1) is None
        attached = db.query(UploadedFile).filter_by(file_id="attached").one()
        assert attached.task_id is None
        assert attached.detached_reason == "task_create_failed"
        assert attached.detached_at is not None
        draft = db.query(UploadedFile).filter_by(file_id="draft").one()
        assert draft.task_id is None
        assert draft.detached_reason is None
        assert draft.detached_at is None


@pytest.mark.parametrize("rebind", [False, True])
def test_durable_replacement_preserves_or_cancels_detachment(sessions, rebind):
    from xagent.web.services.uploaded_file_store import (
        StagedUploadedFile,
        UploadedFileStore,
        snapshot_uploaded_file_version,
    )

    with sessions() as db:
        row = db.query(UploadedFile).filter_by(file_id="attached").one()
        row.storage_status = "available"
        row.storage_key = "users/1/uploads/attached/version-1"
        row.checksum = "verified"
        db.commit()
        purge_task_rows(db, task_id=1, detached_reason="task_deleted")
        db.commit()
        db.refresh(row)
        old = snapshot_uploaded_file_version(row)
        staged = replace(
            StagedUploadedFile.from_record(row), task_id=2 if rebind else None
        )
        receipt = UploadedFileStore(db).upsert_already_durable(
            staged, expected=old, allow_task_rebind=rebind
        )
        db.commit()
        db.refresh(row)
        assert receipt.snapshot == snapshot_uploaded_file_version(row)
        assert row.detached_reason == (None if rebind else "task_deleted")
        assert row.detached_at == (None if rebind else old.detached_at)


@pytest.mark.parametrize("operation", ["insert", "update", "restore"])
@pytest.mark.parametrize("target", ["missing", "foreign"])
def test_metadata_write_rejects_missing_or_foreign_attachment_task(
    sessions, operation, target
):
    from xagent.web.services.uploaded_file_store import (
        StagedUploadedFile,
        UploadedFileStore,
        UploadedFileVersionConflict,
        snapshot_uploaded_file_version,
    )

    with sessions() as db:
        foreign = User(username="foreign", password_hash="unused")
        db.add(foreign)
        db.flush()
        db.add(Task(id=3, user_id=foreign.id, title="foreign"))
        row = db.query(UploadedFile).filter_by(file_id="draft").one()
        row.storage_status = "available"
        row.storage_backend = "file"
        row.storage_key = "users/1/uploads/draft/draft.txt"
        row.checksum = "draft"
        db.commit()
        expected = snapshot_uploaded_file_version(row)
        task_id = 999 if target == "missing" else 3
        staged = replace(StagedUploadedFile.from_record(row), task_id=task_id)
        if operation == "insert":
            staged = replace(
                staged,
                file_id=f"new-{target}",
                storage_key=f"users/1/uploads/new-{target}/file.txt",
            )

        store = UploadedFileStore(db)
        with pytest.raises(
            UploadedFileVersionConflict, match="Attachment task no longer exists"
        ):
            if operation == "restore":
                store.restore_metadata_version_no_commit(
                    expected=expected,
                    replacement=replace(expected, task_id=task_id),
                )
            else:
                store.upsert_already_durable(
                    staged,
                    expected=None if operation == "insert" else expected,
                    allow_task_rebind=operation == "update",
                )

        db.expire_all()
        assert db.query(UploadedFile).filter_by(file_id=f"new-{target}").first() is None
        assert db.query(UploadedFile).filter_by(file_id="draft").one().task_id is None


@pytest.mark.parametrize(
    "dry_run,conversation_days", [(False, 1), (True, 1), (False, 3650)]
)
def test_retention_marks_only_committed_conversation_expiry(
    sessions, dry_run, conversation_days
):
    from xagent.web.models.task import TraceEvent
    from xagent.web.services.task_retention_purge import purge_task

    now = datetime.now(UTC)
    with sessions() as db:
        task = db.get(Task, 1)
        task.last_activity_at = now - timedelta(days=100)
        db.add(
            TraceEvent(
                task_id=1,
                event_id="retention",
                event_type="test",
                timestamp=now,
                data={},
            )
        )
        db.commit()
        purge_task(
            db,
            1,
            now=now,
            conversation_days=conversation_days,
            trace_days=30,
            dry_run=dry_run,
        )
        db.expire_all()
        row = db.query(UploadedFile).filter_by(file_id="attached").one()
        deleted = not dry_run and conversation_days == 1
        assert row.task_id == (None if deleted else 1)
        assert row.detached_reason == ("task_deleted" if deleted else None)
        assert (row.detached_at is not None) == deleted


def test_task_config_cannot_implicitly_recover_a_detached_reference(sessions):
    from xagent.web.services.task_command_execution import (
        SELECTED_FILE_IDS_AGENT_CONFIG_KEY,
        _selected_file_refs_from_task,
    )

    with sessions() as db:
        purge_task_rows(db, task_id=1, detached_reason="task_deleted")
        db.commit()
        task = db.get(Task, 2)
        task.agent_config = {SELECTED_FILE_IDS_AGENT_CONFIG_KEY: ["attached", "draft"]}
        refs = _selected_file_refs_from_task(task, db)
        assert [ref["file_id"] for ref in refs] == ["draft"]
        assert (
            bind_turn_files_no_commit(
                db=db, file_ids=["attached"], task_id=2, owner_user_id=1
            )
            == []
        )
        db.commit()
        assert {ref["file_id"] for ref in _selected_file_refs_from_task(task, db)} == {
            "attached",
            "draft",
        }


@pytest.mark.parametrize("cached", [False, True])
def test_workspace_registration_requires_explicit_rebind_of_retained_file(
    sessions, tmp_path, monkeypatch, cached
):
    from xagent.core.workspace import TaskWorkspace

    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "durable").as_uri())
    path = tmp_path / "attached.txt"
    path.write_text("retained bytes")
    workspace = TaskWorkspace(
        "task_2",
        base_dir=str(tmp_path / "workspaces"),
        allowed_external_dirs=[str(tmp_path)],
        db_task_id=2,
    )
    workspace.owner_user_id = 1
    if cached:
        workspace._recently_registered_files[str(path)] = "attached"
    with sessions() as db:
        purge_task_rows(db, task_id=1, detached_reason="task_deleted")
        db.commit()
        with pytest.raises(ValueError, match="explicitly reattached"):
            workspace.register_file(str(path), db_session=db)
        row = db.query(UploadedFile).filter_by(file_id="attached").one()
        assert row.task_id is None
        assert row.detached_reason == "task_deleted"
        assert path.read_text() == "retained bytes"
        db.rollback()
        assert (
            bind_turn_files_no_commit(
                db=db, file_ids=["attached"], task_id=2, owner_user_id=1
            )
            == []
        )
        db.commit()
        assert workspace.register_file(str(path), db_session=db) == "attached"
