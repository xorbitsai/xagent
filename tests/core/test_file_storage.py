import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

import xagent.core.file_storage.factory as file_storage_factory
from xagent.core.file_storage.factory import get_file_storage
from xagent.core.file_storage.storage import FsspecFileStorage


def test_local_file_storage_round_trips_file(monkeypatch, tmp_path):
    storage_root = tmp_path / "objects"
    materialize_dir = tmp_path / "materialized"
    source = tmp_path / "source.txt"
    source.write_text("hello durable storage", encoding="utf-8")

    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", storage_root.as_uri())
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(materialize_dir))
    get_file_storage.cache_clear()

    storage = get_file_storage()
    stored = storage.put_file(
        source, "users/1/uploads/file-id/source.txt", "text/plain"
    )

    assert stored.backend == "file"
    assert stored.key == "users/1/uploads/file-id/source.txt"
    assert stored.size == len("hello durable storage")
    assert storage.exists(stored.key)

    with storage.open_read(stored.key) as handle:
        assert handle.read() == b"hello durable storage"

    materialized = storage.materialize(stored.key, "source.txt")
    assert materialized.is_relative_to(materialize_dir)
    assert materialized.name == "source.txt"
    assert materialized.read_text(encoding="utf-8") == "hello durable storage"

    listed = storage.list("users/1/uploads")
    assert [item.key for item in listed] == [stored.key]

    storage.delete(stored.key)
    assert not storage.exists(stored.key)


def test_s3_file_storage_uses_bounded_client_timeouts(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummyStorage:
        pass

    def fake_url_to_fs(uri: str, **options: object):
        captured["uri"] = uri
        captured["options"] = options
        return DummyStorage(), "bucket/root"

    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", "s3://bucket/root")
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path))
    monkeypatch.delenv("XAGENT_FILE_STORAGE_OPTIONS", raising=False)
    monkeypatch.setattr(file_storage_factory.fsspec.core, "url_to_fs", fake_url_to_fs)
    get_file_storage.cache_clear()

    storage = get_file_storage()

    assert storage._backend == "s3"
    assert captured["options"] == {
        "config_kwargs": {
            "connect_timeout": 3,
            "read_timeout": 10,
            "retries": {"max_attempts": 1},
        }
    }


def test_s3_file_storage_keeps_explicit_client_timeout_overrides(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class DummyStorage:
        pass

    def fake_url_to_fs(uri: str, **options: object):
        captured["options"] = options
        return DummyStorage(), "bucket/root"

    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", "s3://bucket/root")
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path))
    monkeypatch.setenv(
        "XAGENT_FILE_STORAGE_OPTIONS",
        json.dumps(
            {
                "endpoint_url": "http://minio:9000",
                "config_kwargs": {
                    "connect_timeout": 1,
                    "read_timeout": 2,
                    "retries": {"max_attempts": 3},
                },
            }
        ),
    )
    monkeypatch.setattr(file_storage_factory.fsspec.core, "url_to_fs", fake_url_to_fs)
    get_file_storage.cache_clear()

    get_file_storage()

    assert captured["options"] == {
        "endpoint_url": "http://minio:9000",
        "config_kwargs": {
            "connect_timeout": 1,
            "read_timeout": 2,
            "retries": {"max_attempts": 3},
        },
    }


def test_put_file_hashes_while_copying(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_file_storage.cache_clear()

    storage = get_file_storage()
    source = tmp_path / "single-pass.txt"
    source.write_bytes(b"hash while copying")

    def fail_second_read(path: Path) -> str:
        raise AssertionError(f"unexpected second read for checksum: {path}")

    monkeypatch.setattr(storage, "_sha256", fail_second_read)

    stored = storage.put_file(source, "uploads/single-pass.txt", "text/plain")

    assert stored.checksum
    assert storage.open_read(stored.key).read() == b"hash while copying"


def test_local_file_storage_put_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_file_storage.cache_clear()

    storage = get_file_storage()
    stored = storage.put_bytes(b"abc", "bytes/data.bin")

    assert stored.size == 3
    assert Path(stored.uri.removeprefix("file://")).read_bytes() == b"abc"


def test_object_uri_quotes_key_without_backend_branch(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_file_storage.cache_clear()

    storage = get_file_storage()
    stored = storage.put_bytes(b"abc", "uploads/file with spaces.txt")

    assert stored.uri.endswith("/uploads/file%20with%20spaces.txt")


def test_local_file_storage_does_not_return_signed_url(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_file_storage.cache_clear()

    storage = get_file_storage()

    assert storage.signed_url("uploads/data.txt", expires=300) is None


def test_s3_signed_url_includes_response_headers(tmp_path):
    class S3UrlStorage:
        def __init__(self):
            self.calls = []

        def url(self, path, **kwargs):
            self.calls.append((path, kwargs))
            return "https://cdn.example.com/signed"

    backend = S3UrlStorage()
    storage = FsspecFileStorage(
        fs=backend,
        root="bucket/root",
        backend="s3",
        base_uri="s3://bucket/root",
        materialize_dir=tmp_path,
    )

    signed_url = storage.signed_url(
        "uploads/data.txt",
        expires=120,
        content_type="text/plain",
        content_disposition="inline; filename*=UTF-8''data.txt",
    )

    assert signed_url == "https://cdn.example.com/signed"
    assert backend.calls == [
        (
            "bucket/root/uploads/data.txt",
            {
                "expires": 120,
                "ResponseContentType": "text/plain",
                "ResponseContentDisposition": "inline; filename*=UTF-8''data.txt",
            },
        )
    ]


def test_s3_signed_url_falls_back_for_basic_fsspec_url_signature(tmp_path):
    class BasicUrlStorage:
        def __init__(self):
            self.calls = []

        def url(self, path, *, expires):
            self.calls.append((path, expires))
            return "https://cdn.example.com/basic"

    backend = BasicUrlStorage()
    storage = FsspecFileStorage(
        fs=backend,
        root="bucket/root",
        backend="s3",
        base_uri="s3://bucket/root",
        materialize_dir=tmp_path,
    )

    signed_url = storage.signed_url(
        "uploads/data.txt",
        expires=120,
        content_type="text/plain",
    )

    assert signed_url == "https://cdn.example.com/basic"
    assert backend.calls == [("bucket/root/uploads/data.txt", 120)]


def test_list_uses_detailed_find_metadata_without_per_object_info(tmp_path):
    class DetailedFindStorage:
        def exists(self, path):
            return True

        def find(self, path, detail=False):
            assert detail is True
            return {
                f"{path}/first.txt": {
                    "type": "file",
                    "size": 5,
                    "ETag": "etag-first",
                },
                f"{path}/nested/second.txt": {
                    "type": "file",
                    "size": 6,
                    "etag": "etag-second",
                },
            }

        def info(self, path):
            raise AssertionError(f"unexpected per-object info call: {path}")

    storage = FsspecFileStorage(
        fs=DetailedFindStorage(),
        root="bucket/root",
        backend="s3",
        base_uri="s3://bucket/root",
        materialize_dir=tmp_path,
    )

    listed = storage.list("users/1/uploads")

    assert [(item.key, item.size, item.etag) for item in listed] == [
        ("users/1/uploads/first.txt", 5, "etag-first"),
        ("users/1/uploads/nested/second.txt", 6, "etag-second"),
    ]


def test_put_file_passes_content_type_to_backend_open(tmp_path):
    class WriteHandle:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def write(self, data):
            return len(data)

    class ContentTypeStorage:
        def __init__(self):
            self.open_kwargs = None

        def open(self, path, mode, **kwargs):
            self.open_kwargs = kwargs
            return WriteHandle()

        def makedirs(self, path, exist_ok=False):
            return None

        def info(self, path):
            return {"size": 7}

    backend = ContentTypeStorage()
    storage = FsspecFileStorage(
        fs=backend,
        root="bucket/root",
        backend="s3",
        base_uri="s3://bucket/root",
        materialize_dir=tmp_path,
    )
    source = tmp_path / "data.txt"
    source.write_text("content", encoding="utf-8")

    storage.put_file(source, "uploads/data.txt", "text/plain")

    assert backend.open_kwargs == {"content_type": "text/plain"}


def test_s3_put_bytes_persists_sha256_metadata(tmp_path):
    class WriteHandle:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def write(self, data):
            return len(data)

    class S3LikeStorage:
        def __init__(self):
            self.copy_kwargs = None

        def open(self, path, mode, **kwargs):
            return WriteHandle()

        def makedirs(self, path, exist_ok=False):
            return None

        def info(self, path):
            return {"size": 4}

        def copy(self, source, destination, **kwargs):
            self.copy_kwargs = kwargs

    backend = S3LikeStorage()
    storage = FsspecFileStorage(
        fs=backend,
        root="bucket/root",
        backend="s3",
        base_uri="s3://bucket/root",
        materialize_dir=tmp_path,
    )

    storage.put_bytes(b"data", "uploads/data.txt", "text/plain")

    assert backend.copy_kwargs == {
        "Metadata": {
            "xagent-sha256": "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7"
        },
        "MetadataDirective": "REPLACE",
        "ContentType": "text/plain",
    }


def test_s3_content_hash_reads_sha256_metadata(tmp_path):
    class S3LikeStorage:
        def split_path(self, path):
            assert path == "bucket/root/uploads/data.txt"
            return "bucket", "root/uploads/data.txt", None

        def call_s3(self, method, **kwargs):
            assert method == "head_object"
            assert kwargs == {"Bucket": "bucket", "Key": "root/uploads/data.txt"}
            return {
                "Metadata": {"xagent-sha256": "sha256-value"},
                "ETag": "etag-value",
            }

        def info(self, path):
            raise AssertionError(f"unexpected info fallback: {path}")

    storage = FsspecFileStorage(
        fs=S3LikeStorage(),
        root="bucket/root",
        backend="s3",
        base_uri="s3://bucket/root",
        materialize_dir=tmp_path,
    )

    assert storage.content_hash("uploads/data.txt") == "sha256-value"


def test_s3_content_hash_reads_native_sha256_checksum(tmp_path):
    class S3LikeStorage:
        def split_path(self, path):
            return "bucket", "root/uploads/data.txt", None

        def call_s3(self, method, **kwargs):
            return {"ChecksumSHA256": "native-sha256"}

        def info(self, path):
            raise AssertionError(f"unexpected info fallback: {path}")

    storage = FsspecFileStorage(
        fs=S3LikeStorage(),
        root="bucket/root",
        backend="s3",
        base_uri="s3://bucket/root",
        materialize_dir=tmp_path,
    )

    assert storage.content_hash("uploads/data.txt") == "native-sha256"


def test_local_file_storage_copies_object_to_path(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_file_storage.cache_clear()

    storage = get_file_storage()
    stored = storage.put_bytes(b"restore me", "copies/data.txt")
    target = tmp_path / "restored" / "data.txt"

    copied = storage.copy_to_path(stored.key, target)

    assert copied == target
    assert target.read_bytes() == b"restore me"


def test_copy_to_path_does_not_publish_partial_file_on_read_failure(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_file_storage.cache_clear()

    storage = get_file_storage()
    target = tmp_path / "restored" / "data.txt"

    class FailingRead:
        def __init__(self):
            self._returned_partial = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self, size=-1):
            if not self._returned_partial:
                self._returned_partial = True
                return b"partial"
            raise OSError("durable read failed")

    monkeypatch.setattr(storage, "open_read", lambda key: FailingRead())

    with pytest.raises(OSError, match="durable read failed"):
        storage.copy_to_path("copies/data.txt", target)

    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_copy_to_path_uses_unique_temp_file_per_attempt(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    get_file_storage.cache_clear()

    storage = get_file_storage()
    stored = storage.put_bytes(b"restore me", "copies/data.txt")
    target = tmp_path / "restored" / "data.txt"
    wait_at_open = threading.Barrier(2)
    resume_reads = threading.Event()

    class BlockingRead:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            wait_at_open.wait(timeout=5)
            resume_reads.wait(timeout=5)
            return self

        def __exit__(self, exc_type, exc, tb):
            self._handle.close()
            return None

        def read(self, size=-1):
            return self._handle.read(size)

    original_open_read = storage.open_read

    def blocking_open_read(key):
        return BlockingRead(original_open_read(key))

    monkeypatch.setattr(storage, "open_read", blocking_open_read)

    errors = []

    def copy_attempt():
        try:
            storage.copy_to_path(stored.key, target)
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=copy_attempt)
    second = threading.Thread(target=copy_attempt)
    first.start()
    second.start()

    deadline = time.monotonic() + 5
    while len(list(target.parent.glob(".data.txt.*.tmp"))) < 2:
        if time.monotonic() > deadline:
            break
        time.sleep(0.01)

    assert len(list(target.parent.glob(".data.txt.*.tmp"))) == 2

    resume_reads.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert errors == []
    assert target.read_bytes() == b"restore me"
    assert list(target.parent.glob(".data.txt.*.tmp")) == []


def test_materialize_isolates_objects_with_same_filename(monkeypatch, tmp_path):
    materialize_dir = tmp_path / "materialized"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(materialize_dir))
    get_file_storage.cache_clear()

    storage = get_file_storage()
    first = storage.put_bytes(b"first content", "users/1/tasks/1/output/report.txt")
    second = storage.put_bytes(b"second content", "users/2/tasks/2/output/report.txt")

    first_path = storage.materialize(first.key, "report.txt")
    second_path = storage.materialize(second.key, "report.txt")

    assert first_path != second_path
    assert first_path.is_relative_to(materialize_dir)
    assert second_path.is_relative_to(materialize_dir)
    assert first_path.name == "report.txt"
    assert second_path.name == "report.txt"
    assert first_path.read_bytes() == b"first content"
    assert second_path.read_bytes() == b"second content"


def test_materialize_reuses_existing_cached_file(monkeypatch, tmp_path):
    materialize_dir = tmp_path / "materialized"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(materialize_dir))
    get_file_storage.cache_clear()

    storage = get_file_storage()
    stored = storage.put_bytes(b"cached content", "users/1/uploads/file.txt")
    first_path = storage.materialize(stored.key, "file.txt")

    def fail_open_read(key):
        raise AssertionError(f"unexpected storage read for cached file: {key}")

    monkeypatch.setattr(storage, "open_read", fail_open_read)

    second_path = storage.materialize(stored.key, "file.txt")

    assert second_path == first_path
    assert second_path.read_bytes() == b"cached content"


def test_materialize_uses_content_hash_in_cache_path(monkeypatch, tmp_path):
    materialize_dir = tmp_path / "materialized"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(materialize_dir))
    get_file_storage.cache_clear()

    storage = get_file_storage()
    content = b"cache identity"
    stored = storage.put_bytes(content, "users/1/uploads/file.txt")
    content_hash = hashlib.sha256(content).hexdigest()

    materialized = storage.materialize(stored.key, "file.txt")

    assert materialized == (
        materialize_dir
        / hashlib.sha256(stored.key.encode("utf-8")).hexdigest()[:16]
        / content_hash
        / "file.txt"
    )
    assert not list(materialize_dir.rglob("*.metadata.json"))


def test_materialize_refreshes_cached_file_when_object_changes(monkeypatch, tmp_path):
    materialize_dir = tmp_path / "materialized"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(materialize_dir))
    get_file_storage.cache_clear()

    storage = get_file_storage()
    stored = storage.put_bytes(b"old-data", "users/1/uploads/file.txt")
    first_path = storage.materialize(stored.key, "file.txt")

    storage.put_bytes(b"new-data", stored.key)
    second_path = storage.materialize(stored.key, "file.txt")

    assert second_path != first_path
    assert second_path.read_bytes() == b"new-data"


def test_content_hash_raises_when_backend_hash_is_unavailable(tmp_path):
    class HashlessStorage:
        def info(self, path):
            return {"size": 4, "type": "file"}

    storage = FsspecFileStorage(
        fs=HashlessStorage(),
        root="bucket/root",
        backend="memory",
        base_uri="memory://bucket/root",
        materialize_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="content hash"):
        storage.content_hash("users/1/uploads/file.txt")
