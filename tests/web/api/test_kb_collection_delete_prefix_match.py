"""Collection delete's directory pass takes rows under that exact directory (#2731)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from tests.web.api import test_kb_orphan_reference_scale as scale
from xagent.core.file_storage import get_unscoped_file_storage
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import kb_collection_service
from xagent.web.services.managed_file_ref import ManagedFileRef

test_env = scale.test_env
temp_uploads = scale.temp_uploads
OTHER = scale.OTHER_TENANT


@pytest.fixture(autouse=True)
def _durable_store(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_unscoped_file_storage.cache_clear()
    yield
    get_unscoped_file_storage.cache_clear()


def _upload(sessions, tmp_path: Path, owner: int, path: Path) -> tuple[str, str]:
    """Add a row at ``path``; its bytes are in the durable store, not at ``path``."""
    staged = tmp_path / "staged" / path.name
    staged.parent.mkdir(exist_ok=True)
    staged.write_text(path.name)
    session = sessions()
    try:
        record = UploadedFile(
            user_id=owner,
            filename=path.name,
            storage_path=str(staged),
            mime_type="text/plain",
            file_size=len(path.name),
        )
        session.add(record)
        session.flush()
        ManagedFileRef(record).sync_to_durable()
        record.storage_path = str(path)  # type: ignore[assignment]
        session.commit()
        return str(record.file_id), str(record.storage_key)
    finally:
        session.close()


def _kept(sessions, file_id: str, key: str) -> tuple[bool, bool]:
    return scale._row_exists(sessions, file_id), get_unscoped_file_storage().exists(key)


@pytest.mark.parametrize("sibling", ["my-kb", "MY_KB"])
def test_sibling_collection_rows_are_kept(test_env, temp_uploads, tmp_path, sibling):
    app, headers, user, sessions = test_env
    user_dir = temp_uploads.resolve() / f"user_{user.id}"
    own = _upload(sessions, tmp_path, user.id, user_dir / "my_kb" / "a.txt")
    theirs = _upload(sessions, tmp_path, user.id, user_dir / sibling / "b.txt")
    scale._documents().add([scale._doc(sibling, "doc-b", theirs[0], user.id)])

    response = scale._delete_via_api(app, headers, "/api/kb/collections/my_kb")

    assert response.status_code == 200, response.text
    assert _kept(sessions, *own) == (False, False)
    assert _kept(sessions, *theirs) == (True, True)


def test_rows_stored_through_a_symlinked_root_are_deleted(
    test_env, temp_uploads, tmp_path
):
    app, headers, user, sessions = test_env
    real_root = (tmp_path / "real-user-root").resolve()
    real_root.mkdir()
    (temp_uploads / f"user_{user.id}").symlink_to(real_root)
    composed = temp_uploads / f"user_{user.id}" / "my_kb" / "a.txt"
    rows = [
        _upload(sessions, tmp_path, user.id, composed),
        _upload(sessions, tmp_path, user.id, real_root / "my_kb" / "b.txt"),
    ]

    response = scale._delete_via_api(app, headers, "/api/kb/collections/my_kb")

    assert response.status_code == 200, response.text
    assert not (real_root / "my_kb").exists()
    assert [_kept(sessions, *row) for row in rows] == [(False, False)] * 2


def test_orphan_rows_in_the_directory_are_still_deleted(
    test_env, temp_uploads, tmp_path
):
    app, headers, user, sessions = test_env
    session = sessions()
    try:
        session.add(User(id=OTHER, username=f"u{OTHER}", password_hash="x"))
        session.commit()
    finally:
        session.close()
    my_kb = temp_uploads.resolve() / f"user_{user.id}" / "my_kb"
    own = [
        _upload(sessions, tmp_path, user.id, my_kb / "a.txt"),
        _upload(sessions, tmp_path, user.id, my_kb),
    ]
    other = _upload(sessions, tmp_path, OTHER, my_kb / "c.txt")

    response = scale._delete_via_api(app, headers, "/api/kb/collections/my_kb")

    assert response.status_code == 200, response.text
    assert [_kept(sessions, *row) for row in own] == [(False, False)] * 2
    assert _kept(sessions, *other) == (True, True)


def test_prefix_match_escapes_like_wildcards(tmp_path):
    engine = create_engine("sqlite://")
    UploadedFile.__table__.create(engine)
    likes: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _capture(_conn, _cursor, statement, *_rest):
        if " LIKE " in statement:
            likes.append(statement)

    with Session(engine) as db:
        kb_collection_service._delete_collection_uploaded_files_impl(
            db,
            user_id=1,
            collection_file_ids=set(),
            remaining_file_ids=set(),
            collection_dir=tmp_path / "my_kb",
            after_commit=[],
        )

    assert likes
    assert all(s.count(" LIKE ") == s.count(" ESCAPE '/'") for s in likes)
