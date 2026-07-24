import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.web.models.database import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.managed_file_ref import DurableStorageOperationError
from xagent.web.services.uploaded_file_store import (
    UploadedFileStore,
    delete_svg_png_cache,
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


def test_sync_existing_refreshes_durable_object(monkeypatch, tmp_path):
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

    source.write_text("new content", encoding="utf-8")
    store.sync_existing(record)
    db.commit()

    with get_unscoped_file_storage().open_read(storage_key) as handle:
        assert handle.read() == b"new content"


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


def test_delete_svg_png_cache_removes_legacy_path_keyed_preview(monkeypatch, tmp_path):
    """Legacy (non-UUID) files have no file_id, so ``_svg_png_cache_path``
    falls back to a bare ``<path-hash>.preview.png`` name with no file_id
    prefix. Passing the resolved source path must find and remove that
    entry too, since the file_id glob alone can never match it."""
    import hashlib

    storage_root = tmp_path / "storage"
    cache_dir = storage_root / "svg_png_cache"
    cache_dir.mkdir(parents=True)

    legacy_source = tmp_path / "legacy" / "notes.svg"
    legacy_source.parent.mkdir(parents=True)
    legacy_source.write_text("<svg></svg>", encoding="utf-8")

    path_key = hashlib.sha256(str(legacy_source.resolve()).encode()).hexdigest()[:24]
    legacy_cache_path = cache_dir / f"{path_key}.preview.png"
    legacy_cache_path.write_bytes(b"legacy-png")

    unrelated_cache = cache_dir / "some-other-hash.preview.png"
    unrelated_cache.write_bytes(b"unrelated-png")

    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(storage_root))

    delete_svg_png_cache(None, source_path=legacy_source)

    assert not legacy_cache_path.exists()
    assert unrelated_cache.exists()


def test_delete_svg_png_cache_uuid_case_unaffected_by_source_path_arg(
    monkeypatch, tmp_path
):
    """A UUID file_id must keep cleaning up via the glob prefix regardless
    of whether a source_path is also supplied (regression guard)."""
    storage_root = tmp_path / "storage"
    cache_dir = storage_root / "svg_png_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "file-svg.aaa111.preview.png").write_bytes(b"png-a")

    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(storage_root))

    delete_svg_png_cache("file-svg", source_path=None)

    assert not list(cache_dir.glob("file-svg.*.preview.png"))


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
    with get_unscoped_file_storage().open_read(first_key) as handle:
        assert handle.read() == b"second"


def test_upsert_by_storage_path_refreshes_same_size_rewrite(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
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
    with get_unscoped_file_storage().open_read(first_key) as handle:
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

    first = store.upsert_by_storage_path(
        user_id=int(user.id),
        filename="kb.md",
        storage_path=source,
        mime_type="text/markdown",
        file_size=source.stat().st_size,
        storage_key="users/1/uploads/file-kb/kb.md",
    )
    first_key = str(first.storage_key)
    db.commit()

    second_key = "users/1/uploads/file-kb-renamed/kb.md"
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
