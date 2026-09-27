"""Collection delete keeps files in its directory that other documents use (#2662)."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.web.api import test_kb_orphan_reference_scale as scale
from xagent.core.file_storage import get_unscoped_file_storage
from xagent.core.tools.core.RAG_tools.core.schemas import CollectionOperationResult
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
    for owner, file_id in ((user.id, mine), (OTHER, theirs)):
        kept = Path(_row(sessions, file_id).storage_path)
        assert not _dir(temp_uploads, owner).exists()
        assert kept.exists()


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
    test_env, temp_uploads, monkeypatch
):
    app, headers, user, sessions = test_env
    team = _dir(temp_uploads, user.id)
    kept = _upload(sessions, user.id, team / "kept.txt")
    loose = _upload(sessions, user.id, team / "loose.txt")
    scale._documents().add([scale._doc("other", "doc-other", kept, user.id)])

    def _raise(_file_ids):
        raise RuntimeError("refs down")

    monkeypatch.setattr(kb_collection_service, "find_referenced_file_ids", _raise)

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "partial_success"
    assert "refs down" in response.json()["message"]
    assert (team / "kept.txt").exists()
    assert _row(sessions, kept).storage_path == str(team / "kept.txt")
    assert _row(sessions, loose) is not None


def test_copy_failure_keeps_the_directory_and_paths(
    test_env, temp_uploads, monkeypatch
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
    real_copy = shutil.copy2
    copied = []

    def _copy(src, dst, *args, **kwargs):
        copied.append(src)
        if len(copied) == 2:
            Path(dst).write_text("par")
            raise OSError("disk full")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(kb_collection_service, "shutil", SimpleNamespace(copy2=_copy))

    response = scale._delete_via_api(app, headers, "/api/kb/collections/team")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "partial_success"
    assert _row(sessions, first).storage_path == str(team / "first.txt")
    assert _row(sessions, second).storage_path == str(team / "second.txt")
    assert (team / "first.txt").exists()
    assert (team / "second.txt").exists()
    assert not list(temp_uploads.rglob(".kb-retained/*/*"))


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
    assert "Could not keep files" in response.json()["message"]
    assert _row(sessions, file_id).storage_path == str(team / "a.txt")
    assert (team / "a.txt").exists()
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

    monkeypatch.setattr(kb_collection_service, "shutil", SimpleNamespace(copy2=_fail))

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
    assert not team.exists()


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
    sibling_path = _dir(temp_uploads, user.id, "my-kb") / "b.txt"
    sibling = _upload(sessions, user.id, sibling_path)
    scale._documents().add(
        [
            scale._doc("other", "doc-a", inside, user.id),
            scale._doc("my-kb", "doc-b", sibling, user.id),
        ]
    )

    db = sessions()
    try:
        result = kb_collection_service.delete_collection_physical_dir(
            db, user_id=user.id, collection_name="my_kb"
        )
    finally:
        db.close()

    assert result.status == "success"
    assert _row(sessions, sibling).storage_path == str(sibling_path)
    assert Path(_row(sessions, inside).storage_path).read_text() == "a.txt"
