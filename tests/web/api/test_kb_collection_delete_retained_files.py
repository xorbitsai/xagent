"""Collection delete keeps files in its directory that other documents use (#2662)."""

from __future__ import annotations

import errno
import io
import logging
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from tests.web.api import test_kb_orphan_reference_scale as scale
from xagent.core.file_storage import get_unscoped_file_storage
from xagent.core.tools.core.RAG_tools.core.schemas import CollectionOperationResult
from xagent.web.api import files as files_module
from xagent.web.api import kb as kb_module
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import kb_collection_service
from xagent.web.services.knowledge_base_team_scope import (
    set_knowledge_base_team_hooks,
    snapshot_knowledge_base_team_hooks,
)
from xagent.web.services.managed_file_ref import ManagedFileRef
from xagent.web.services.uploaded_file_store import UploadedFileStore

test_env = scale.test_env
temp_uploads = scale.temp_uploads
OTHER = scale.OTHER_TENANT


@pytest.fixture(autouse=True)
def _durable_store(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    yield
    get_unscoped_file_storage.cache_clear()


def _dir(temp_uploads: Path, owner: int, collection: str = "team") -> Path:
    return temp_uploads.resolve() / f"user_{owner}" / collection


def _upload(sessions, owner: int, path: Path, *, durable: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(path.name)
    session = sessions()
    try:
        record = UploadedFile(
            user_id=owner,
            filename=path.name,
            storage_path=str(path),
            mime_type="text/plain",
            file_size=len(path.name),
        )
        session.add(record)
        session.flush()
        if durable:
            ManagedFileRef(record).sync_to_durable()
        session.commit()
        return str(record.file_id)
    finally:
        session.close()


def _row(sessions, file_id: str) -> UploadedFile | None:
    session = sessions()
    try:
        return session.query(UploadedFile).filter_by(file_id=file_id).first()
    finally:
        session.close()


def _add_user(sessions, user_id: int) -> None:
    session = sessions()
    try:
        session.add(User(id=user_id, username=f"u{user_id}", password_hash="x"))
        session.commit()
    finally:
        session.close()


def _refuse_link(*_args, **_kwargs):
    raise OSError(errno.EXDEV, "Invalid cross-device link")


def _patch_link_and_copy(monkeypatch, *, link, copy) -> None:
    monkeypatch.setattr(
        kb_collection_service, "os", SimpleNamespace(sep=os.sep, link=link)
    )
    monkeypatch.setattr(kb_collection_service, "shutil", SimpleNamespace(copy2=copy))


@pytest.mark.parametrize("linkable", [True, False], ids=["hard-link", "copy-fallback"])
def test_retained_file_is_linked_or_copied(
    test_env, temp_uploads, monkeypatch, linkable
):
    app, headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    file_id = _upload(sessions, user.id, team / "a.txt")
    scale._documents().add([scale._doc("other", "doc-other", file_id, user.id)])
    if not linkable:
        _patch_link_and_copy(monkeypatch, link=_refuse_link, copy=shutil.copy2)

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.json()["status"] == "success"
    kept = Path(_row(sessions, file_id).storage_path)
    assert team not in kept.parents
    assert kept.read_text() == "a.txt"
    assert kept.stat().st_nlink == 1
    # A linked file's old name is dropped before the move; a copy's original moves.
    assert bool(list((temp_uploads / ".trash").rglob("a.txt"))) is not linkable


def test_failed_move_after_the_commit_cannot_write_into_the_retained_file(
    test_env, temp_uploads, monkeypatch
):
    _app, _headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    file_id = _upload(sessions, user.id, team / "a.txt")
    scale._documents().add([scale._doc("other", "doc-other", file_id, user.id)])

    def _refuse_move(*_args, **_kwargs):
        raise OSError("rename refused")

    monkeypatch.setattr(
        kb_collection_service, "move_collection_dir_to_trash", _refuse_move
    )
    db = sessions()
    try:
        result = kb_collection_service.delete_collection_physical_dir(
            db, user_id=user.id, collection_name="team"
        )
    finally:
        db.close()
    kept = Path(_row(sessions, file_id).storage_path)
    kb_module._copy_upload_file_to_path(
        SimpleNamespace(file=io.BytesIO(b"NEW")), team / "a.txt"
    )

    assert result.status == "failed"
    assert team not in kept.parents
    assert kept.read_text() == "a.txt"


def test_retained_file_keeps_its_durable_object(test_env, temp_uploads):
    app, headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    file_id = _upload(sessions, user.id, team / "a.txt", durable=True)
    key = str(_row(sessions, file_id).storage_key)
    scale._documents().add(
        [
            scale._doc("team", "doc-team", file_id, user.id),
            scale._doc("other", "doc-other", file_id, user.id),
        ]
    )

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert get_unscoped_file_storage().exists(key)
    record = _row(sessions, file_id)
    kept = Path(record.storage_path)
    kept.unlink()
    assert UploadedFileStore.ensure_local(record) == kept
    assert not team.exists()


def test_file_only_referenced_outside_the_collection_is_kept(test_env, temp_uploads):
    app, headers, user, sessions = test_env
    file_id = _upload(sessions, user.id, _dir(temp_uploads, user.id) / "a.txt")
    scale._documents().add([scale._doc("other", "doc-other", file_id, user.id)])

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert Path(_row(sessions, file_id).storage_path).read_text() == "a.txt"


def test_unreferenced_files_in_the_directory_are_still_deleted(test_env, temp_uploads):
    app, headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    kept = _upload(sessions, user.id, team / "kept.txt", durable=True)
    orphan = _upload(sessions, user.id, team / "orphan.txt", durable=True)
    loose = _upload(sessions, user.id, team / "loose.txt", durable=True)
    orphan_key = str(_row(sessions, orphan).storage_key)
    loose_key = str(_row(sessions, loose).storage_key)
    scale._documents().add(
        [
            scale._doc("team", "doc-kept", kept, user.id),
            scale._doc("team", "doc-orphan", orphan, user.id),
            scale._doc("other", "doc-other", kept, user.id),
        ]
    )

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert _row(sessions, kept) is not None
    assert _row(sessions, orphan) is None
    assert _row(sessions, loose) is None
    assert not get_unscoped_file_storage().exists(orphan_key)
    assert not get_unscoped_file_storage().exists(loose_key)
    assert not (temp_uploads / f"user_{user.id}" / ".kb-retained" / orphan).exists()


def test_admin_delete_keeps_each_owners_referenced_file(test_env, temp_uploads):
    app, headers, user, sessions = test_env
    _add_user(sessions, OTHER)
    session = sessions()
    session.get(User, user.id).is_admin = True
    session.commit()
    session.close()
    mine = _upload(sessions, user.id, _dir(temp_uploads, user.id) / "mine.txt")
    theirs = _upload(sessions, OTHER, _dir(temp_uploads, OTHER) / "theirs.txt")
    scale._documents().add(
        [
            scale._doc("team", "doc-mine", mine, user.id),
            scale._doc("team", "doc-theirs", theirs, OTHER),
            scale._doc("elsewhere", "copy-mine", mine, OTHER),
            scale._doc("elsewhere", "copy-theirs", theirs, user.id),
        ]
    )

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    for owner, file_id, name in (
        (user.id, mine, "mine.txt"),
        (OTHER, theirs, "theirs.txt"),
    ):
        kept = Path(_row(sessions, file_id).storage_path)
        assert not _dir(temp_uploads, owner).exists()
        assert _dir(temp_uploads, owner).parent in kept.resolve().parents
        assert kept.read_text() == name


def test_another_owners_row_in_the_directory_moves_under_its_owner(
    test_env, temp_uploads
):
    app, headers, user, sessions = test_env
    _add_user(sessions, OTHER)
    file_id = _upload(sessions, OTHER, _dir(temp_uploads, user.id) / "legacy.txt")
    scale._documents().add([scale._doc("theirs", "doc-theirs", file_id, OTHER)])

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    kept = Path(_row(sessions, file_id).storage_path)
    assert (temp_uploads / f"user_{OTHER}").resolve() in kept.resolve().parents
    assert kept.read_text() == "legacy.txt"


def test_reference_lookup_failure_keeps_the_directory(
    test_env, temp_uploads, monkeypatch, caplog
):
    app, headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    kept = _upload(sessions, user.id, team / "kept.txt")
    loose = _upload(sessions, user.id, team / "loose.txt")
    scale._documents().add([scale._doc("other", "doc-other", kept, user.id)])

    def _raise(_file_ids):
        raise RuntimeError("refs down")

    monkeypatch.setattr(kb_collection_service, "find_referenced_file_ids", _raise)

    with caplog.at_level(logging.WARNING, logger=kb_collection_service.__name__):
        response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "partial_success"
    assert "Could not check which files" in response.json()["message"]
    assert "The directory was kept." in response.json()["message"]
    assert ".." not in response.json()["message"]
    assert "refs down" not in response.json()["message"]
    assert any(
        record.exc_info and "refs down" in str(record.exc_info[1].__cause__)
        for record in caplog.records
    )
    assert (team / "kept.txt").exists()
    assert _row(sessions, kept).storage_path == str(team / "kept.txt")
    assert _row(sessions, loose) is not None


@pytest.mark.parametrize("linkable", [True, False], ids=["linked", "copied"])
def test_copy_failure_keeps_the_directory_and_paths(
    test_env, temp_uploads, monkeypatch, linkable
):
    app, headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    first = _upload(sessions, user.id, team / "first.txt")
    second = _upload(sessions, user.id, team / "second.txt")
    scale._documents().add(
        [
            scale._doc("other", "doc-1", first, user.id),
            scale._doc("other", "doc-2", second, user.id),
        ]
    )
    attempts = []

    def _link(src, dst):
        attempts.append(src)
        if linkable and len(attempts) == 1:
            return os.link(src, dst)
        return _refuse_link()

    def _copy(src, dst, *args, **kwargs):
        if len(attempts) == 2:
            Path(dst).write_text("par")
            raise OSError("disk full")
        return shutil.copy2(src, dst, *args, **kwargs)

    _patch_link_and_copy(monkeypatch, link=_link, copy=_copy)

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "partial_success"
    assert "Could not copy out files" in response.json()["message"]
    assert "disk full" not in response.json()["message"]
    for file_id, name in ((first, "first.txt"), (second, "second.txt")):
        assert _row(sessions, file_id).storage_path == str(team / name)
        assert (team / name).read_text() == name
    assert not list(temp_uploads.rglob(".kb-retained/*"))


def test_commit_failure_keeps_the_directory_and_paths(test_env, temp_uploads):
    app, headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    file_id = _upload(sessions, user.id, team / "a.txt")
    retained = temp_uploads / f"user_{user.id}" / ".kb-retained" / file_id / "a.txt"
    _upload(sessions, user.id, retained)
    retained.unlink()
    scale._documents().add([scale._doc("other", "doc-other", file_id, user.id)])

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "partial_success"
    assert "Could not commit before moving" in response.json()["message"]
    assert "UNIQUE" not in response.json()["message"]
    assert _row(sessions, file_id).storage_path == str(team / "a.txt")
    assert (team / "a.txt").exists()
    assert retained.read_text() == "a.txt"

    session = sessions()
    session.query(UploadedFile).filter_by(storage_path=str(retained)).delete()
    session.commit()
    session.close()
    retry = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert retry.json()["status"] == "success"
    assert _row(sessions, file_id).storage_path == str(retained)
    assert retained.read_text() == "a.txt"


def _delete_team_dir_with_pending_delete(sessions, user_id: int, pending: str):
    db = sessions()
    try:
        db.delete(db.query(UploadedFile).filter_by(file_id=pending).one())
        db.flush()
        result = kb_collection_service.delete_collection_physical_dir(
            db, user_id=user_id, collection_name="team"
        )
        still_pending = db.query(UploadedFile).filter_by(file_id=pending).count() == 0
        db.rollback()
        return result, still_pending
    finally:
        db.close()


def test_failed_copy_keeps_the_callers_pending_writes(
    test_env, temp_uploads, monkeypatch
):
    _app, _headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    file_id = _upload(sessions, user.id, team / "a.txt")
    pending = _upload(sessions, user.id, temp_uploads / f"user_{user.id}" / "p.txt")
    scale._documents().add([scale._doc("other", "doc-other", file_id, user.id)])

    def _fail(*_args, **_kwargs):
        raise OSError("disk full")

    _patch_link_and_copy(monkeypatch, link=_refuse_link, copy=_fail)

    result, still_pending = _delete_team_dir_with_pending_delete(
        sessions, user.id, pending
    )

    assert result.status == "failed"
    assert still_pending
    assert (team / "a.txt").exists()


def test_directory_without_referenced_files_commits_nothing(test_env, temp_uploads):
    _app, _headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    _upload(sessions, user.id, team / "a.txt")
    pending = _upload(sessions, user.id, temp_uploads / f"user_{user.id}" / "p.txt")

    result, _still_pending = _delete_team_dir_with_pending_delete(
        sessions, user.id, pending
    )

    assert result.status == "success"
    assert not team.exists()
    assert _row(sessions, pending) is not None


def test_referenced_row_without_a_local_file_is_repointed(test_env, temp_uploads):
    app, headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    file_id = _upload(sessions, user.id, team / "a.txt", durable=True)
    (team / "a.txt").unlink()
    scale._documents().add([scale._doc("other", "doc-other", file_id, user.id)])

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "success"
    assert UploadedFileStore.ensure_local(_row(sessions, file_id)).read_text() == (
        "a.txt"
    )
    assert not team.exists()


@pytest.mark.parametrize("local", [True, False], ids=["local-copy", "durable-only"])
def test_retained_path_survives_a_later_failure(
    test_env, temp_uploads, monkeypatch, local
):
    app, headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    file_id = _upload(sessions, user.id, team / "a.txt", durable=True)
    if not local:
        (team / "a.txt").unlink()
    scale._documents().add(
        [
            scale._doc("team", "doc-team", file_id, user.id),
            scale._doc("other", "doc-other", file_id, user.id),
        ]
    )

    def _raise(_file_ids):
        raise RuntimeError("refs down")

    monkeypatch.setattr(kb_module, "_find_referenced_file_ids", _raise)

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 500
    assert not team.exists()
    record = _row(sessions, file_id)
    assert team not in Path(record.storage_path).parents
    assert UploadedFileStore.ensure_local(record).read_text() == "a.txt"


def test_failed_collection_delete_leaves_directory_rows_and_hook_writes(
    test_env, temp_uploads
):
    app, headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    file_ids = [_upload(sessions, user.id, team / f"f{i}.txt") for i in range(2)]
    link = _upload(sessions, user.id, temp_uploads / f"user_{user.id}" / "link.txt")
    scale._documents().add(
        [scale._doc("team", f"doc-{i}", f, user.id) for i, f in enumerate(file_ids)]
    )

    def _hook(db, _user_id, _name, _unused):
        db.delete(db.query(UploadedFile).filter_by(file_id=link).one())
        db.flush()

    def _fail(collection, _user_id, _is_admin):
        return CollectionOperationResult(
            status="error", collection=collection, message="listing failed"
        )

    with (
        snapshot_knowledge_base_team_hooks(),
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch("xagent.web.api.kb.delete_collection", side_effect=_fail),
    ):
        set_knowledge_base_team_hooks(deleted=_hook)
        response = TestClient(app).delete("/api/kb/collections/team", headers=headers)

    assert response.json()["status"] == "error"
    for i, file_id in enumerate(file_ids):
        assert _row(sessions, file_id).storage_path == str(team / f"f{i}.txt")
        assert (team / f"f{i}.txt").exists()
    assert _row(sessions, link) is not None


def test_last_reference_frees_another_owners_retained_file(test_env, temp_uploads):
    app, headers, user, sessions = test_env
    _add_user(sessions, OTHER)
    file_id = _upload(
        sessions, OTHER, _dir(temp_uploads, OTHER) / "a.txt", durable=True
    )
    key = str(_row(sessions, file_id).storage_key)
    scale._documents().add([scale._doc("mine", "doc-mine", file_id, user.id)])
    db = sessions()
    try:
        kb_collection_service.delete_collection_physical_dir(
            db, user_id=OTHER, collection_name="team"
        )
    finally:
        db.close()
    kept = Path(_row(sessions, file_id).storage_path)
    assert kept.read_text() == "a.txt"

    response = scale._delete_via_api(
        app, headers, f"/api/kb/collections/mine/documents/a.txt?file_id={file_id}"
    )

    assert response.json()["deleted_doc_ids"] == ["doc-mine"]
    assert _row(sessions, file_id) is None
    assert not get_unscoped_file_storage().exists(key)
    assert not kept.exists()


@pytest.mark.parametrize(
    "layout", ["file-id-dir-outside-kb-retained", "kb-retained-other-id"]
)
def test_another_owners_lookalike_row_is_not_freed(test_env, temp_uploads, layout):
    app, headers, user, sessions = test_env
    _add_user(sessions, OTHER)
    root = temp_uploads / f"user_{OTHER}"
    file_id = _upload(sessions, OTHER, root / "tmp" / "x.txt")
    parent = (
        root / "uploads" / file_id
        if layout == "file-id-dir-outside-kb-retained"
        else root / ".kb-retained" / f"not-{file_id}"
    )
    path = parent / "x.txt"
    path.parent.mkdir(parents=True)
    path.write_text("theirs")
    session = sessions()
    session.query(UploadedFile).filter_by(file_id=file_id).update(
        {"storage_path": str(path)}
    )
    session.commit()
    session.close()
    scale._documents().add([scale._doc("mine", "doc-mine", file_id, user.id)])

    response = scale._delete_via_api(
        app, headers, f"/api/kb/collections/mine/documents/x.txt?file_id={file_id}"
    )

    assert response.json()["deleted_doc_ids"] == ["doc-mine"]
    assert _row(sessions, file_id) is not None
    assert path.read_text() == "theirs"


def test_backfill_mints_no_row_for_retained_copies(test_env, temp_uploads, monkeypatch):
    _app, _headers, user, sessions = test_env
    monkeypatch.setattr(files_module, "get_uploads_dir", lambda: temp_uploads)
    root = temp_uploads / f"user_{user.id}"
    leftover = root / ".kb-retained" / "f-1" / "a.txt"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("a")
    retained = _upload(sessions, user.id, root / ".kb-retained" / "f-2" / "b.txt")
    visible = [root / ".notes.pdf", root / "task_2" / "f" / ".draft.docx"]
    for path in visible:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)

    db = sessions()
    try:
        files_module._backfill_uploaded_file_records(db, db.get(User, user.id))
        paths = {row.storage_path for row in db.query(UploadedFile)}
    finally:
        db.close()

    assert str(leftover) not in paths
    assert {str(path) for path in visible} <= paths
    assert _row(sessions, retained).storage_key is not None


async def test_whole_collection_rollback_keeps_referenced_file(
    test_env, temp_uploads, monkeypatch
):
    _app, _headers, user, sessions = test_env
    fresh = _dir(temp_uploads, user.id, "fresh")
    own_path = fresh / "own.txt"
    own = _upload(sessions, user.id, own_path)
    shared = _upload(sessions, user.id, fresh / "shared.txt")
    scale._documents().add(
        [
            scale._doc("fresh", "doc-fresh", own, user.id),
            scale._doc("other", "doc-other", shared, user.id),
        ]
    )
    scale._stub_rollback_leaves(monkeypatch, may_delete=True)

    await scale._rollback(
        kb_module._rollback_failed_ingestion,
        sessions,
        user,
        file_id=own,
        path=own_path,
        collection="fresh",
        collection_existed_before=False,
    )

    assert _row(sessions, own) is None
    assert Path(_row(sessions, shared).storage_path).read_text() == "shared.txt"


def test_files_stored_under_either_spelling_are_kept(test_env, temp_uploads, tmp_path):
    app, headers, user, sessions = test_env
    real_root = tmp_path / "real-user-root"
    real_root.mkdir()
    (temp_uploads / f"user_{user.id}").symlink_to(real_root)
    composed = _upload(
        sessions, user.id, temp_uploads / f"user_{user.id}" / "team" / "a.txt"
    )
    resolved = _upload(sessions, user.id, real_root.resolve() / "team" / "b.txt")
    scale._documents().add(
        [
            scale._doc("other", "doc-a", composed, user.id),
            scale._doc("other", "doc-b", resolved, user.id),
        ]
    )

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert not (real_root / "team").exists()
    assert Path(_row(sessions, composed).storage_path).read_text() == "a.txt"
    assert Path(_row(sessions, resolved).storage_path).read_text() == "b.txt"


def test_only_rows_inside_the_directory_are_retained(test_env, temp_uploads):
    _app, _headers, user, sessions = test_env
    inside = _upload(sessions, user.id, _dir(temp_uploads, user.id, "my_kb") / "a.txt")
    siblings = {
        _upload(sessions, user.id, path): path
        for path in (
            _dir(temp_uploads, user.id, "my-kb") / "b.txt",
            _dir(temp_uploads, user.id, "MY_KB") / "c.txt",
        )
    }
    scale._documents().add(
        [scale._doc("other", "doc-a", inside, user.id)]
        + [scale._doc("sib", f"doc-{f}", f, user.id) for f in siblings]
    )
    statements: list[str] = []

    def _record(_conn, _cursor, statement, *_args):
        statements.append(statement)

    db = sessions()
    event.listen(db.get_bind(), "before_cursor_execute", _record)
    try:
        result = kb_collection_service.delete_collection_physical_dir(
            db, user_id=user.id, collection_name="my_kb"
        )
    finally:
        event.remove(db.get_bind(), "before_cursor_execute", _record)
        db.close()

    assert result.status == "success"
    for file_id, path in siblings.items():
        assert _row(sessions, file_id).storage_path == str(path)
    assert Path(_row(sessions, inside).storage_path).read_text() == "a.txt"
    assert sum(s.count("ESCAPE") for s in statements if "uploaded_files" in s) == 2
