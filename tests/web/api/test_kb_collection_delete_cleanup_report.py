"""How collection delete reports each owner's directory cleanup (#2665)."""

from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.web.api import test_kb_dir as kb_dir
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionOperationDetail,
    CollectionOperationResult,
)
from xagent.web.api import kb as kb_module
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.kb_collection_service import (
    CollectionPhysicalDeleteResult as Cleanup,
)
from xagent.web.services.kb_collection_service import (
    classify_collection_physical_cleanup,
)

test_env = kb_dir.test_env
temp_uploads = kb_dir.temp_uploads

DELETED = "Collection 'demo' deleted successfully."
PARTIAL = "Partially deleted collection 'demo'."


def _db_result(status: str = "success", message: str = DELETED, warnings=("db",)):
    return CollectionOperationResult(
        status=status,
        # The route reports the sanitized name, not this one.
        collection="from-db",
        message=message,
        warnings=list(warnings),
        affected_documents=[CollectionOperationDetail(doc_id="d1", status="success")],
        deleted_counts={"chunks": 3},
    )


def _expected(db_result, status: str, message: str, warnings: list[str]):
    return CollectionOperationResult(
        status=status,
        collection="demo",
        message=message,
        warnings=warnings,
        affected_documents=db_result.affected_documents,
        deleted_counts=db_result.deleted_counts,
    )


def _dir(temp_uploads: Path, owner: int) -> Path:
    return temp_uploads / f"user_{owner}" / "demo"


def _delete(test_env, temp_uploads, cleanups: dict[int, Cleanup], db_result):
    """Admin-delete ``demo``, whose owners are found by their rows in its dir."""
    sessions = test_env[3]
    db = sessions()
    try:
        for owner in cleanups:
            path = _dir(temp_uploads, owner) / f"{owner}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
            db.add(
                UploadedFile(
                    user_id=owner,
                    filename=path.name,
                    storage_path=str(path.resolve()),
                    mime_type="text/plain",
                    file_size=1,
                )
            )
        db.commit()
        store = MagicMock()
        store.list_document_records.return_value = []
        physical = MagicMock(side_effect=lambda _db, **kw: cleanups[kw["user_id"]])
        with ExitStack() as stack:
            for target, kwargs in (
                ("xagent.web.api.kb.get_vector_index_store", {"return_value": store}),
                (
                    "xagent.core.tools.core.RAG_tools.storage.factory."
                    "get_vector_index_store",
                    {"return_value": store},
                ),
                ("xagent.web.api.kb.delete_collection", {"return_value": db_result}),
                ("xagent.web.api.kb.delete_collection_physical_dir", {"new": physical}),
            ):
                stack.enter_context(patch(target, **kwargs))
            result = kb_module._perform_kb_collection_delete(
                "demo", int(test_env[2].id), True, db
            )
        db.commit()
        kept = {int(r.user_id) for r in db.query(UploadedFile).all()}
    finally:
        db.close()
    return result, physical, kept


def _moved(temp_uploads: Path, owner: int) -> str:
    return (
        f"Physical directory moved to trash for user_{owner}: "
        f"{_dir(temp_uploads, owner)} "
        "(trash cleanup requires external scheduler/cron)"
    )


def _not_found(owner: int) -> str:
    return (
        f"Physical directory cleanup for user_{owner}: "
        "No physical directory found (collection had no files)"
    )


def _uncertain(owner: int, error: str) -> str:
    return (
        f"Physical directory cleanup for user_{owner}: Warning - {error}. "
        "Database deletion proceeded, but physical file cleanup status is uncertain."
    )


def _failed(owner: int, error: str) -> str:
    return f"Physical directory cleanup for user_{owner}: Failed - {error}"


@pytest.mark.parametrize("dir_reported", [True, False])
def test_moved_directory_is_noted_and_its_rows_deleted(
    test_env, temp_uploads, dir_reported
):
    owner = int(test_env[2].id)
    # Without a reported dir, the route falls back to the owner's upload path.
    cleanup = Cleanup(
        "success", collection_dir=_dir(temp_uploads, owner) if dir_reported else None
    )
    db_result = _db_result()

    result, _, kept = _delete(test_env, temp_uploads, {owner: cleanup}, db_result)

    note = _moved(temp_uploads, owner)
    assert result == _expected(db_result, "success", f"{DELETED} {note}.", ["db", note])
    assert kept == set()


@pytest.mark.parametrize(
    ("cleanup", "status", "note", "rows_kept"),
    [
        (Cleanup("not_found"), "success", _not_found, False),
        (
            Cleanup("error", error="bad path"),
            "partial_success",
            lambda o: _uncertain(o, "bad path"),
            True,
        ),
        (
            Cleanup("failed", error="locked"),
            "partial_success",
            lambda o: _failed(o, "locked"),
            True,
        ),
        (Cleanup("error"), "success", None, True),
        (Cleanup("failed"), "success", None, True),
        (Cleanup("skipped", error="odd"), "success", None, True),
    ],
    ids=[
        "not_found",
        "error",
        "failed",
        "error-no-msg",
        "failed-no-msg",
        "other-status",
    ],
)
def test_each_cleanup_status_is_reported(
    test_env, temp_uploads, cleanup, status, note, rows_kept
):
    owner = int(test_env[2].id)
    db_result = _db_result()

    result, _, kept = _delete(test_env, temp_uploads, {owner: cleanup}, db_result)

    if note is None:
        assert result == _expected(db_result, status, DELETED, ["db"])
    else:
        text = note(owner)
        assert result == _expected(
            db_result, status, f"{DELETED} {text}.", ["db", text]
        )
    assert kept == ({owner} if rows_kept else set())


def test_owners_are_noted_in_id_order_and_only_clean_ones_lose_rows(
    test_env, temp_uploads, caplog
):
    first = int(test_env[2].id)
    cleanups = {
        first + 3: Cleanup("failed", error="locked"),
        first + 2: Cleanup("success", collection_dir=_dir(temp_uploads, first + 2)),
        first: Cleanup("not_found"),
        first + 1: Cleanup("error", error="bad path"),
    }
    db_result = _db_result(warnings=("db-1", "db-2"))

    with caplog.at_level(logging.WARNING, logger=kb_module.logger.name):
        result, physical, kept = _delete(test_env, temp_uploads, cleanups, db_result)

    notes = [
        _not_found(first),
        _uncertain(first + 1, "bad path"),
        _moved(temp_uploads, first + 2),
        _failed(first + 3, "locked"),
    ]
    assert [c.kwargs["user_id"] for c in physical.call_args_list] == sorted(cleanups)
    assert result == _expected(
        db_result,
        "partial_success",
        f"{DELETED} {'; '.join(notes)}.",
        ["db-1", "db-2", *notes],
    )
    assert kept == {first + 1, first + 3}
    assert [
        r.getMessage() for r in caplog.records if "Preserving" in r.getMessage()
    ] == [
        f"Preserving UploadedFile records for collection demo/user_{owner} "
        f"because physical cleanup status is {status}"
        for owner, status in ((first + 1, "error"), (first + 3, "failed"))
    ]


def test_clean_owners_alone_leave_a_successful_delete_as_success(
    test_env, temp_uploads
):
    first = int(test_env[2].id)
    cleanups = {
        first: Cleanup("success", collection_dir=_dir(temp_uploads, first)),
        first + 1: Cleanup("not_found"),
        first + 2: Cleanup("failed"),
    }
    db_result = _db_result(warnings=())

    result, _, kept = _delete(test_env, temp_uploads, cleanups, db_result)

    notes = [_moved(temp_uploads, first), _not_found(first + 1)]
    assert result == _expected(
        db_result, "success", f"{DELETED} {'; '.join(notes)}.", notes
    )
    assert kept == {first + 2}


@pytest.mark.parametrize("db_status", ["partial_success", "unexpected"])
def test_only_a_successful_db_delete_is_downgraded(test_env, temp_uploads, db_status):
    owner = int(test_env[2].id)
    db_result = _db_result(status=db_status, message=PARTIAL)

    result, _, kept = _delete(
        test_env, temp_uploads, {owner: Cleanup("failed", error="locked")}, db_result
    )

    note = _failed(owner, "locked")
    assert result == _expected(db_result, db_status, f"{PARTIAL} {note}.", ["db", note])
    assert kept == {owner}


def test_failed_db_delete_skips_directory_cleanup_and_keeps_rows(
    test_env, temp_uploads
):
    owner = int(test_env[2].id)
    db_result = _db_result(status="error", message="Failed to delete collection.")

    result, physical, kept = _delete(
        test_env, temp_uploads, {owner: Cleanup("success")}, db_result
    )

    physical.assert_not_called()
    assert result == _expected(db_result, "error", db_result.message, ["db"])
    assert kept == {owner}


def test_classifier_rejects_a_failed_db_delete():
    with pytest.raises(ValueError):
        classify_collection_physical_cleanup(
            _db_result(status="error"), {1: Cleanup("success")}, collection_name="demo"
        )
