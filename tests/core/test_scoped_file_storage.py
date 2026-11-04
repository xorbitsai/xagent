import pytest

from xagent.core.file_storage import (
    ScopedFileStorage,
    StorageKeyScopeError,
    get_unscoped_file_storage,
    get_user_file_storage,
)
from xagent.core.file_storage.storage import FsspecFileStorage


@pytest.fixture
def local_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized"))
    get_unscoped_file_storage.cache_clear()
    yield get_unscoped_file_storage()
    get_unscoped_file_storage.cache_clear()


def test_user_scoped_round_trip(local_storage, tmp_path):
    storage = get_user_file_storage(1)
    source = tmp_path / "source.txt"
    source.write_text("scoped content", encoding="utf-8")

    stored = storage.put_file(source, "users/1/uploads/file-id/source.txt")
    assert stored.key == "users/1/uploads/file-id/source.txt"
    assert storage.exists(stored.key)
    assert storage.stat(stored.key).size == len("scoped content")

    with storage.open_read(stored.key) as handle:
        assert handle.read() == b"scoped content"

    assert [item.key for item in storage.list("users/1/uploads")] == [stored.key]

    materialized = storage.materialize(stored.key, "source.txt")
    assert materialized.read_bytes() == b"scoped content"

    copied = storage.copy_to_path(stored.key, tmp_path / "restored" / "source.txt")
    assert copied.read_bytes() == b"scoped content"

    assert storage.content_hash(stored.key) == stored.checksum

    storage.delete(stored.key)
    assert not storage.exists(stored.key)


def test_every_operation_rejects_out_of_scope_keys(local_storage, tmp_path):
    storage = get_user_file_storage(1)
    source = tmp_path / "source.txt"
    source.write_text("data", encoding="utf-8")
    foreign_key = "users/2/uploads/file-id/source.txt"

    with pytest.raises(StorageKeyScopeError):
        storage.put_file(source, foreign_key)
    with pytest.raises(StorageKeyScopeError):
        storage.put_bytes(b"data", foreign_key)
    with pytest.raises(StorageKeyScopeError):
        storage.open_read(foreign_key)
    with pytest.raises(StorageKeyScopeError):
        storage.exists(foreign_key)
    with pytest.raises(StorageKeyScopeError):
        storage.stat(foreign_key)
    with pytest.raises(StorageKeyScopeError):
        storage.content_hash(foreign_key)
    with pytest.raises(StorageKeyScopeError):
        storage.list("users/2/uploads")
    with pytest.raises(StorageKeyScopeError):
        storage.delete(foreign_key)
    with pytest.raises(StorageKeyScopeError):
        storage.materialize(foreign_key)
    with pytest.raises(StorageKeyScopeError):
        storage.copy_to_path(foreign_key, tmp_path / "restored" / "x.txt")
    with pytest.raises(StorageKeyScopeError):
        storage.signed_url(foreign_key, expires=300)


def test_prefix_containment_is_separator_aware(local_storage):
    storage = get_user_file_storage(1)

    with pytest.raises(StorageKeyScopeError):
        storage.exists("users/10/uploads/file-id/source.txt")
    with pytest.raises(StorageKeyScopeError):
        storage.list("users/10")


def test_key_equal_to_prefix_is_in_scope(local_storage):
    storage = get_user_file_storage(1)

    assert storage.list("users/1") == []
    assert storage.exists("users/1") is False


def test_structural_violations_raise_value_error_not_scope_error(
    local_storage, tmp_path
):
    storage = get_user_file_storage(1)

    with pytest.raises(ValueError):
        storage.put_bytes(b"data", "users/1/../2/escape.txt")
    with pytest.raises(ValueError):
        storage.open_read("users/1/uploads/../../2/escape.txt")
    with pytest.raises(ValueError):
        storage.put_bytes(b"data", "users/1/uploads/id/back\\slash.txt")


def test_scoped_read_tolerates_in_scope_legacy_keys(local_storage, tmp_path):
    storage = get_user_file_storage(1)
    legacy_key = "users/1/uploads/id/back\\slash.txt"
    legacy_object = tmp_path / "objects" / "users/1/uploads/id" / "back\\slash.txt"
    legacy_object.parent.mkdir(parents=True)
    legacy_object.write_bytes(b"legacy data")

    with storage.open_read(legacy_key) as handle:
        assert handle.read() == b"legacy data"

    storage.delete(legacy_key)
    assert not storage.exists(legacy_key)


def test_scoped_read_rejects_out_of_scope_legacy_keys(local_storage):
    storage = get_user_file_storage(1)

    with pytest.raises(StorageKeyScopeError):
        storage.open_read("users/2/uploads/id/back\\slash.txt")


def test_signed_url_scope_enforcement_on_s3_backend(tmp_path):
    class S3UrlStorage:
        def __init__(self):
            self.calls = []

        def url(self, path, **kwargs):
            self.calls.append((path, kwargs))
            return "https://cdn.example.com/signed"

    backend = S3UrlStorage()
    storage = ScopedFileStorage(
        storage=FsspecFileStorage(
            fs=backend,
            root="bucket/root",
            backend="s3",
            base_uri="s3://bucket/root",
            materialize_dir=tmp_path,
        ),
        prefix="users/1",
    )

    signed = storage.signed_url("users/1/uploads/id/data.txt", expires=120)
    assert signed == "https://cdn.example.com/signed"
    assert backend.calls[0][0] == "bucket/root/users/1/uploads/id/data.txt"

    with pytest.raises(StorageKeyScopeError):
        storage.signed_url("users/2/uploads/id/data.txt", expires=120)
    with pytest.raises(StorageKeyScopeError):
        storage.signed_url("users/10/uploads/id/data.txt", expires=120)
    assert len(backend.calls) == 1


def test_get_user_file_storage_binds_expected_prefix(local_storage):
    assert get_user_file_storage(42).prefix == "users/42"


def test_get_user_file_storage_empty_segments_is_owner_prefix(local_storage):
    assert get_user_file_storage(42, scope_segments=()).prefix == "users/42"


def test_get_user_file_storage_binds_scoped_prefix(local_storage):
    handle = get_user_file_storage(
        42, scope_segments=("clients", "3", "end_users", "7")
    )
    assert handle.prefix == "users/42/clients/3/end_users/7"


def test_scoped_handle_admits_owner_key_and_rejects_sibling(local_storage, tmp_path):
    # A deeper prefix stays a strict extension of the owner root: the scope's
    # own key round-trips, a sibling scope's key under the same owner does not.
    handle = get_user_file_storage(1, scope_segments=("clients", "3", "end_users", "7"))
    source = tmp_path / "source.txt"
    source.write_text("data", encoding="utf-8")

    own_key = "users/1/clients/3/end_users/7/uploads/file-id/source.txt"
    assert handle.put_file(source, own_key).key == own_key

    with pytest.raises(StorageKeyScopeError):
        handle.put_file(
            source, "users/1/clients/3/end_users/8/uploads/file-id/source.txt"
        )
