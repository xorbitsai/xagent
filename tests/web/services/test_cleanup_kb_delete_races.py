"""KB deletion yields to an uploaded-file compensation claim."""

from __future__ import annotations

from datetime import UTC, datetime
from threading import Event, Thread
from typing import Callable

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.test_complete_upload_cleanup import (
    lifecycle as cleanup_lifecycle_fixture,
)
from xagent.core.file_storage import get_unscoped_file_storage
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.kb_collection_service import (
    _delete_collection_uploaded_files_impl,
)
from xagent.web.services.uploaded_file_store import UploadedFileStore

lifecycle = cleanup_lifecycle_fixture


def _claim_after_lookup(
    *,
    engine: Engine,
    sessions: sessionmaker[Session],
    file_id: str,
    delete: Callable[[Session, list[tuple[str, Callable[[], None]]]], int],
) -> tuple[int, list[tuple[str, Callable[[], None]]]]:
    lookup_started = Event()
    claim_committed = Event()
    armed = True
    result: dict[str, object] = {}

    def pause_after_uploaded_file_lookup(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ) -> None:
        nonlocal armed
        if not armed or not statement.lstrip().upper().startswith("SELECT"):
            return
        if "uploaded_files" not in statement:
            return
        armed = False
        lookup_started.set()
        assert claim_committed.wait(5), "cleanup claim did not commit"

    event_name = (
        "before_cursor_execute"
        if engine.dialect.name == "sqlite"
        else "after_cursor_execute"
    )
    event.listen(engine, event_name, pause_after_uploaded_file_lookup)

    def run_delete() -> None:
        callbacks: list[tuple[str, Callable[[], None]]] = []
        try:
            with sessions() as db:
                result["deleted"] = delete(db, callbacks)
                db.commit()
        except BaseException as exc:  # surfaced in the test thread
            result["error"] = exc
        finally:
            result["callbacks"] = callbacks

    worker = Thread(target=run_delete)
    worker.start()
    try:
        assert lookup_started.wait(5), "KB delete did not inspect the upload"
        with sessions.begin() as claimant:
            claimed = (
                claimant.query(UploadedFile)
                .filter(
                    UploadedFile.file_id == file_id,
                    UploadedFile.storage_status == "available",
                )
                .update(
                    {
                        UploadedFile.storage_status: "compensating",
                        UploadedFile.updated_at: datetime.now(UTC),
                        UploadedFile.cleanup_manifest: {
                            "version": 1,
                            "claim": "test-cleanup",
                        },
                    },
                    synchronize_session=False,
                )
            )
            assert claimed == 1
    finally:
        claim_committed.set()
        worker.join(10)
        event.remove(engine, event_name, pause_after_uploaded_file_lookup)
    assert not worker.is_alive()
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return int(result["deleted"]), result["callbacks"]  # type: ignore[return-value]


@pytest.mark.parametrize("candidate_source", ["document", "directory"])
def test_collection_delete_yields_to_cleanup_claim_without_queuing_resources(
    lifecycle, candidate_source
) -> None:
    sessions, source, materialized, previews, storage_key, file_id = lifecycle
    engine = sessions.kw["bind"]

    def delete(db: Session, callbacks) -> int:
        return _delete_collection_uploaded_files_impl(
            db,
            user_id=1,
            collection_file_ids=(
                {file_id} if candidate_source == "document" else set()
            ),
            remaining_file_ids=set(),
            collection_dir=(source.parent if candidate_source == "directory" else None),
            after_commit=callbacks,
        )

    deleted, callbacks = _claim_after_lookup(
        engine=engine,
        sessions=sessions,
        file_id=file_id,
        delete=delete,
    )

    assert deleted == 0
    assert callbacks == []
    with sessions() as db:
        record = db.query(UploadedFile).filter_by(file_id=file_id).one()
        assert record.storage_status == "compensating"
        assert record.cleanup_manifest == {"version": 1, "claim": "test-cleanup"}
    assert source.exists()
    assert materialized.exists()
    assert get_unscoped_file_storage().exists(storage_key)
    assert all(preview.exists() for preview in previews)


def test_store_defers_preview_cleanup_until_after_commit(lifecycle) -> None:
    sessions, _source, _materialized, previews, _storage_key, file_id = lifecycle
    callbacks: list[tuple[str, Callable[[], None]]] = []

    with sessions() as db:
        record = db.query(UploadedFile).filter_by(file_id=file_id).one()
        UploadedFileStore(db).delete(record, after_commit=callbacks)
        assert all(preview.exists() for preview in previews)
        db.commit()
        assert all(preview.exists() for preview in previews)

    for _label, callback in callbacks:
        callback()
    assert not any(preview.exists() for preview in previews)
