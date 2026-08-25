from hashlib import sha256
from pathlib import Path

import pytest

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.file_storage.scoped import ScopedFileStorage, StorageKeyScopeError
from xagent.core.file_storage.types import StoredObject
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.managed_file_ref import (
    DurableObjectIntegrityError,
    DurableStorageOperationError,
    ManagedFileRef,
)


def _configure_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized"))
    get_unscoped_file_storage.cache_clear()


def _record(local_path, **overrides):
    values = {
        "file_id": "file-123",
        "user_id": 7,
        "filename": local_path.name,
        "storage_path": str(local_path),
        "storage_status": "legacy",
        "mime_type": "text/plain",
        "file_size": 0,
    }
    values.update(overrides)
    return UploadedFile(**values)


def test_ensure_local_returns_existing_local_file(tmp_path):
    source = tmp_path / "uploads" / "local.txt"
    source.parent.mkdir()
    source.write_text("local content", encoding="utf-8")
    record = _record(source)

    assert ManagedFileRef(record).ensure_local() == source


def test_ensure_local_restores_missing_file_from_durable_storage(monkeypatch, tmp_path):
    _configure_storage(monkeypatch, tmp_path)
    storage = get_unscoped_file_storage()
    stored = storage.put_bytes(b"durable content", "users/7/uploads/file-123/local.txt")
    local_path = tmp_path / "uploads" / "local.txt"
    record = _record(
        local_path,
        storage_backend=stored.backend,
        storage_key=stored.key,
        storage_uri=stored.uri,
        storage_status="available",
        checksum=stored.checksum,
    )

    restored = ManagedFileRef(record).ensure_local()

    assert restored == local_path
    assert restored.read_text(encoding="utf-8") == "durable content"


def test_materialize_uses_temp_dir_when_original_path_is_missing(monkeypatch, tmp_path):
    _configure_storage(monkeypatch, tmp_path)
    storage = get_unscoped_file_storage()
    stored = storage.put_bytes(
        b"preview content", "users/7/uploads/file-123/preview.txt"
    )
    local_path = tmp_path / "uploads" / "preview.txt"
    record = _record(
        local_path,
        storage_backend=stored.backend,
        storage_key=stored.key,
        storage_uri=stored.uri,
        storage_status="available",
        checksum=stored.checksum,
    )

    materialized = ManagedFileRef(record).materialize()

    assert materialized.is_relative_to(tmp_path / "materialized")
    assert materialized.name == "preview.txt"
    assert materialized.read_text(encoding="utf-8") == "preview content"
    assert not local_path.exists()


def test_materialize_can_ignore_existing_local_path(monkeypatch, tmp_path):
    _configure_storage(monkeypatch, tmp_path)
    storage = get_unscoped_file_storage()
    stored = storage.put_bytes(
        b"durable content", "users/7/uploads/file-123/preview.txt"
    )
    local_path = tmp_path / "stale-worktree" / "preview.txt"
    local_path.parent.mkdir()
    local_path.write_text("stale local content", encoding="utf-8")
    record = _record(
        local_path,
        storage_backend=stored.backend,
        storage_key=stored.key,
        storage_uri=stored.uri,
        storage_status="available",
        checksum=stored.checksum,
    )

    materialized = ManagedFileRef(record).materialize(allow_existing_local=False)

    assert materialized.is_relative_to(tmp_path / "materialized")
    assert materialized.read_text(encoding="utf-8") == "durable content"
    assert local_path.read_text(encoding="utf-8") == "stale local content"


def test_ensure_local_rejects_restored_checksum_mismatch(monkeypatch, tmp_path):
    _configure_storage(monkeypatch, tmp_path)
    storage = get_unscoped_file_storage()
    stored = storage.put_bytes(
        b"wrong durable content", "users/7/uploads/file-123/bad.txt"
    )
    local_path = tmp_path / "uploads" / "bad.txt"
    record = _record(
        local_path,
        storage_backend=stored.backend,
        storage_key=stored.key,
        storage_uri=stored.uri,
        storage_status="available",
        checksum=sha256(b"expected content").hexdigest(),
    )

    with pytest.raises(DurableObjectIntegrityError, match="re-upload"):
        ManagedFileRef(record).ensure_local()

    assert not local_path.exists()
    assert not list(local_path.parent.glob(f".{local_path.name}.*.tmp"))


def test_materialize_rejects_checksum_mismatch_and_discards_cache(
    monkeypatch, tmp_path
):
    _configure_storage(monkeypatch, tmp_path)
    storage = get_unscoped_file_storage()
    stored = storage.put_bytes(
        b"wrong preview content", "users/7/uploads/file-123/bad-preview.txt"
    )
    local_path = tmp_path / "uploads" / "bad-preview.txt"
    record = _record(
        local_path,
        storage_backend=stored.backend,
        storage_key=stored.key,
        storage_uri=stored.uri,
        storage_status="available",
        checksum=sha256(b"expected preview content").hexdigest(),
    )

    with pytest.raises(DurableObjectIntegrityError, match="re-upload"):
        ManagedFileRef(record).materialize()

    materialized_files = [
        path for path in (tmp_path / "materialized").rglob("*") if path.is_file()
    ]
    assert materialized_files == []
    assert not local_path.exists()


def test_materialize_retries_once_after_discarding_bad_cache(monkeypatch, tmp_path):
    _configure_storage(monkeypatch, tmp_path)
    storage = get_unscoped_file_storage()
    stored = storage.put_bytes(
        b"correct preview content", "users/7/uploads/file-123/cached-preview.txt"
    )
    local_path = tmp_path / "uploads" / "cached-preview.txt"
    record = _record(
        local_path,
        storage_backend=stored.backend,
        storage_key=stored.key,
        storage_uri=stored.uri,
        storage_status="available",
        checksum=stored.checksum,
    )
    cached_path = storage.materialize(stored.key, "cached-preview.txt")
    cached_path.write_bytes(b"stale cached bytes")

    materialized = ManagedFileRef(record).materialize()

    assert materialized == cached_path
    assert materialized.read_bytes() == b"correct preview content"
    assert not local_path.exists()


def test_open_read_restores_and_validates_durable_when_local_missing(
    monkeypatch, tmp_path
):
    _configure_storage(monkeypatch, tmp_path)
    storage = get_unscoped_file_storage()
    stored = storage.put_bytes(b"stream me", "users/7/uploads/file-123/stream.txt")
    local_path = tmp_path / "uploads" / "stream.txt"
    record = _record(
        local_path,
        storage_backend=stored.backend,
        storage_key=stored.key,
        storage_uri=stored.uri,
        storage_status="available",
        checksum=stored.checksum,
    )

    with ManagedFileRef(record).open_read() as handle:
        assert handle.read() == b"stream me"
    assert local_path.read_bytes() == b"stream me"


def test_open_read_prefers_existing_local_file_over_durable(monkeypatch, tmp_path):
    _configure_storage(monkeypatch, tmp_path)
    storage = get_unscoped_file_storage()
    stored = storage.put_bytes(
        b"stale durable content", "users/7/uploads/file-123/current.txt"
    )
    local_path = tmp_path / "uploads" / "current.txt"
    local_path.parent.mkdir()
    local_path.write_bytes(b"current local content")
    record = _record(
        local_path,
        storage_backend=stored.backend,
        storage_key=stored.key,
        storage_uri=stored.uri,
        storage_status="available",
    )

    with ManagedFileRef(record).open_read() as handle:
        assert handle.read() == b"current local content"


def test_signed_access_url_returns_none_without_durable_object(tmp_path):
    record = _record(tmp_path / "uploads" / "local.txt")

    assert ManagedFileRef(record).signed_access_url(expires=300) is None


def test_signed_access_url_delegates_to_storage(tmp_path):
    checksum = sha256(b"remote content").hexdigest()

    class SigningStorage:
        def __init__(self):
            self.calls = []
            self.content_hash_calls = []

        def content_hash(self, key):
            self.content_hash_calls.append(key)
            return checksum

        def signed_url(
            self,
            key,
            *,
            expires,
            content_type=None,
            content_disposition=None,
        ):
            self.calls.append((key, expires, content_type, content_disposition))
            return "https://cdn.example.com/signed"

    storage = SigningStorage()
    record = _record(
        tmp_path / "uploads" / "object.txt",
        storage_backend="s3",
        storage_key="users/7/uploads/file-123/object.txt",
        storage_status="available",
        checksum=checksum,
    )

    signed_url = ManagedFileRef(record, storage=storage).signed_access_url(
        expires=60,
        content_type="text/plain",
        content_disposition="inline",
    )

    assert signed_url == "https://cdn.example.com/signed"
    assert storage.content_hash_calls == ["users/7/uploads/file-123/object.txt"]
    assert storage.calls == [
        ("users/7/uploads/file-123/object.txt", 60, "text/plain", "inline")
    ]


def test_signed_access_url_rejects_checksum_mismatch(tmp_path):
    class MismatchedStorage:
        def content_hash(self, key):
            del key
            return sha256(b"wrong content").hexdigest()

        def signed_url(self, **kwargs):
            raise AssertionError("signed URL should not be generated")

    record = _record(
        tmp_path / "uploads" / "object.txt",
        storage_backend="s3",
        storage_key="users/7/uploads/file-123/object.txt",
        storage_status="available",
        checksum=sha256(b"expected content").hexdigest(),
    )

    with pytest.raises(DurableObjectIntegrityError):
        ManagedFileRef(record, storage=MismatchedStorage()).signed_access_url(
            expires=60
        )


def test_signed_access_url_falls_back_when_checksum_unavailable(tmp_path):
    class UnhashableStorage:
        def __init__(self):
            self.signed_calls = []

        def content_hash(self, key):
            del key
            raise RuntimeError("head metadata unavailable")

        def signed_url(self, **kwargs):
            self.signed_calls.append(kwargs)
            return "https://cdn.example.com/signed"

    storage = UnhashableStorage()
    record = _record(
        tmp_path / "uploads" / "object.txt",
        storage_backend="s3",
        storage_key="users/7/uploads/file-123/object.txt",
        storage_status="available",
        checksum=sha256(b"expected content").hexdigest(),
    )

    assert ManagedFileRef(record, storage=storage).signed_access_url(expires=60) is None
    assert storage.signed_calls == []


def test_sync_to_durable_uploads_local_file_and_updates_record(monkeypatch, tmp_path):
    _configure_storage(monkeypatch, tmp_path)
    source = tmp_path / "uploads" / "sync.txt"
    source.parent.mkdir()
    source.write_text("sync content", encoding="utf-8")
    record = _record(source, file_size=source.stat().st_size)

    stored = ManagedFileRef(record).sync_to_durable()

    assert stored.key == "users/7/uploads/file-123/sync.txt"
    assert record.storage_backend == "file"
    assert record.storage_key == stored.key
    assert record.storage_uri == stored.uri
    assert record.checksum is not None
    assert record.storage_status == "available"
    assert record.file_size == len("sync content")
    with get_unscoped_file_storage().open_read(stored.key) as handle:
        assert handle.read() == b"sync content"


def test_sync_to_durable_accepts_custom_storage_key(monkeypatch, tmp_path):
    _configure_storage(monkeypatch, tmp_path)
    source = tmp_path / "workspace" / "output" / "report.txt"
    source.parent.mkdir(parents=True)
    source.write_text("report", encoding="utf-8")
    record = _record(source, file_id="file-output")

    stored = ManagedFileRef(record).sync_to_durable(
        storage_key="users/7/tasks/42/outputs/file-output/output/report.txt"
    )

    assert stored.key == "users/7/tasks/42/outputs/file-output/output/report.txt"
    assert record.storage_key == stored.key


def test_apply_stored_object_rejects_missing_checksum(tmp_path):
    record = _record(tmp_path / "uploads" / "object.txt")
    stored_object = StoredObject(
        backend="s3",
        key="users/7/uploads/file-123/object.txt",
        uri="s3://bucket/users/7/uploads/file-123/object.txt",
        size=12,
        checksum=None,
    )

    with pytest.raises(ValueError, match="checksum"):
        ManagedFileRef(record).apply_stored_object(stored_object)

    assert record.storage_status == "legacy"
    assert record.storage_key is None


def test_missing_local_and_missing_durable_key_raises(tmp_path):
    record = _record(tmp_path / "missing.txt")

    with pytest.raises(FileNotFoundError):
        ManagedFileRef(record).ensure_local()


class FailingStorage:
    def put_file(self, source, key, content_type=None):
        raise RuntimeError("remote write unavailable")

    def copy_to_path(self, key, target_path):
        raise RuntimeError("remote read unavailable")

    def materialize(self, key, filename=None):
        raise RuntimeError("remote preview unavailable")


class ZeroSizeStorage:
    def __init__(self):
        self.stat_calls: list[str] = []
        self.put_calls: list[tuple[Path, str]] = []

    def stat(self, key):
        self.stat_calls.append(key)
        return StoredObject(
            backend="s3",
            key=key,
            uri=f"s3://bucket/{key}",
            size=0,
            checksum="remote-zero",
            etag="etag",
        )

    def put_file(self, source, key, content_type=None):
        del content_type
        self.put_calls.append((source, key))
        return StoredObject(
            backend="s3",
            key=key,
            uri=f"s3://bucket/{key}",
            size=source.stat().st_size,
            checksum="refreshed",
            etag="etag",
        )

    def content_hash(self, key):
        return f"hash:{key}"


class SameSizeStaleStorage:
    def __init__(self):
        self.stat_calls: list[str] = []
        self.put_calls: list[tuple[Path, str]] = []

    def stat(self, key):
        self.stat_calls.append(key)
        return StoredObject(
            backend="s3",
            key=key,
            uri=f"s3://bucket/{key}",
            size=len(b"new-data"),
            checksum="remote-old-checksum",
            etag="old-etag",
        )

    def put_file(self, source, key, content_type=None):
        del content_type
        self.put_calls.append((source, key))
        return StoredObject(
            backend="s3",
            key=key,
            uri=f"s3://bucket/{key}",
            size=source.stat().st_size,
            checksum=sha256(source.read_bytes()).hexdigest(),
            etag="new-etag",
        )


def test_sync_to_durable_wraps_remote_write_failure(tmp_path):
    """A backend fault genuinely is a durable-storage operation error.

    Pairs with
    ``test_sync_to_durable_propagates_scope_violation_unwrapped``: excluding
    the namespace-authority types from that wrap must not turn it into
    "propagate everything".
    """
    source = tmp_path / "uploads" / "sync-fails.txt"
    source.parent.mkdir()
    source.write_text("sync content", encoding="utf-8")
    record = _record(source, file_size=source.stat().st_size)

    with pytest.raises(DurableStorageOperationError, match="write durable object"):
        ManagedFileRef(record, storage=FailingStorage()).sync_to_durable()

    assert record.storage_status == "legacy"
    assert record.storage_key is None


def test_sync_to_durable_propagates_scope_violation_unwrapped(monkeypatch, tmp_path):
    """A key outside the bound prefix stays a ``StorageKeyScopeError``.

    Folding it into ``DurableStorageOperationError`` would report a permanent
    containment violation as the retryable storage outage that error means to
    callers.
    """
    _configure_storage(monkeypatch, tmp_path)
    source = tmp_path / "uploads" / "foreign-key.txt"
    source.parent.mkdir()
    source.write_text("scope content", encoding="utf-8")
    record = _record(source, file_size=source.stat().st_size)
    storage = ScopedFileStorage(
        storage=get_unscoped_file_storage(), prefix="users/7/clients/3"
    )
    foreign_key = "users/7/clients/4/uploads/file-123/foreign-key.txt"

    with pytest.raises(StorageKeyScopeError) as excinfo:
        ManagedFileRef(record, storage=storage).sync_to_durable(storage_key=foreign_key)

    assert not isinstance(excinfo.value, DurableStorageOperationError)
    assert record.storage_status == "legacy"
    assert record.storage_key is None
    assert not get_unscoped_file_storage().exists(foreign_key)


def test_ensure_local_wraps_remote_read_failure_when_local_missing(tmp_path):
    local_path = tmp_path / "uploads" / "missing-local.txt"
    record = _record(
        local_path,
        storage_backend="s3",
        storage_key="users/7/uploads/file-123/missing-local.txt",
        storage_status="available",
    )

    with pytest.raises(DurableStorageOperationError, match="restore durable object"):
        ManagedFileRef(record, storage=FailingStorage()).ensure_local()


def test_materialize_wraps_remote_preview_failure_when_local_missing(tmp_path):
    local_path = tmp_path / "uploads" / "missing-preview.txt"
    record = _record(
        local_path,
        storage_backend="s3",
        storage_key="users/7/uploads/file-123/missing-preview.txt",
        storage_status="available",
    )

    with pytest.raises(
        DurableStorageOperationError, match="materialize durable object"
    ):
        ManagedFileRef(record, storage=FailingStorage()).materialize()


def test_adopt_existing_object_refreshes_zero_size_remote_object_from_local_file(
    tmp_path,
):
    local_path = tmp_path / "uploads" / "payload.txt"
    local_path.parent.mkdir()
    local_path.write_text("payload", encoding="utf-8")
    record = _record(
        local_path,
        storage_backend="s3",
        storage_key="users/7/uploads/file-123/payload.txt",
        storage_status="legacy",
    )

    storage = ZeroSizeStorage()
    result = ManagedFileRef(record, storage=storage).adopt_existing_object(
        record.storage_key
    )

    assert result == "uploaded"
    assert storage.stat_calls == [record.storage_key]
    assert storage.put_calls == [(local_path, record.storage_key)]
    assert record.checksum == "refreshed"


def test_adopt_existing_object_refreshes_same_size_checksum_mismatch_from_local_file(
    tmp_path,
):
    local_path = tmp_path / "uploads" / "payload.txt"
    local_path.parent.mkdir()
    local_path.write_bytes(b"new-data")
    record = _record(
        local_path,
        storage_backend="s3",
        storage_key="users/7/uploads/file-123/payload.txt",
        storage_status="legacy",
    )

    storage = SameSizeStaleStorage()
    result = ManagedFileRef(record, storage=storage).adopt_existing_object(
        record.storage_key
    )

    assert result == "uploaded"
    assert storage.stat_calls == [record.storage_key]
    assert storage.put_calls == [(local_path, record.storage_key)]
    assert record.checksum == sha256(b"new-data").hexdigest()


def test_the_wrap_keeps_the_storage_key_out_of_its_own_message(tmp_path):
    """``str(exc)`` is the value that escapes; it must not carry the key.

    The key's scope segments encode the owning user's id, and ``str(exc)`` on
    these classes reaches places the raise site does not control: a bare
    ``raise`` from a WebSocket fault arm carries it into a task-wide broadcast
    and a persisted command row, and broad ``except RuntimeError`` arms
    interpolate it into client-facing text (#1497). The invariant therefore
    has to hold at the exception, not at each egress.
    """
    source = tmp_path / "uploads" / "leak-probe.txt"
    source.parent.mkdir()
    source.write_text("leak probe", encoding="utf-8")
    record = _record(source, file_size=source.stat().st_size)

    with pytest.raises(DurableStorageOperationError) as raised:
        ManagedFileRef(record, storage=FailingStorage()).sync_to_durable()

    fault = raised.value
    assert fault.storage_key, "the wrap must carry the key on the attribute"
    assert fault.storage_key not in str(fault)
    assert "users/" not in str(fault)


def test_no_wrap_site_interpolates_into_the_message():
    """Every construction of these classes, not just the one driven above.

    A new wrap site that puts the key back into the message would reopen the
    leak at every ``str(exc)`` egress at once, and no runtime test can drive a
    site that does not exist yet. Parsing the module is what closes that gap.

    It also asserts ``storage_key=`` is present at every site: the invariant
    is the key off the message *and* on the attribute, and mypy enforcing the
    required keyword elsewhere is not a reason to leave the second half
    unpinned here.

    The message must be a string literal or a bare name (a module constant
    like ``FILE_INTEGRITY_REUPLOAD_MESSAGE``) -- a whitelist, because the
    first version of this test blacklisted f-strings only, and ``"..." + key``,
    ``"%s" % key``, and ``"{}".format(key)`` all walked past it. The second
    version read only the positional slot, so ``message=...`` slipped past as
    a keyword; it is looked up in both places now, and a ``**`` unpacking --
    which no static check can see through -- fails outright rather than being
    skipped. Calls are matched by bare name and by attribute
    (``module.DurableStorageOperationError(...)``), with the guarded name set
    derived from the class hierarchy so a future subclass is covered without
    editing this test. What the whitelist still cannot see: a name bound to
    dynamically built text one statement earlier, or a construction hidden
    behind an alias -- which is what the exact count below turns into a
    failure instead of a silent skip.
    """
    import ast

    from xagent.web.services import managed_file_ref

    def _hierarchy_names(cls: type) -> set[str]:
        names = {cls.__name__}
        for sub in cls.__subclasses__():
            names |= _hierarchy_names(sub)
        return names

    guarded = _hierarchy_names(DurableStorageOperationError)

    tree = ast.parse(Path(managed_file_ref.__file__).read_text(encoding="utf-8"))
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        callee_name = (
            callee.id
            if isinstance(callee, ast.Name)
            else callee.attr
            if isinstance(callee, ast.Attribute)
            else None
        )
        if callee_name not in guarded:
            continue
        assert not any(kw.arg is None for kw in node.keywords), (
            f"managed_file_ref.py:{node.lineno} constructs via ** unpacking, "
            "which this check cannot see through; spell the arguments out so "
            "the message stays statically checkable"
        )
        # The invariant has two halves: the key is off the message AND on the
        # attribute. mypy enforces the required keyword, but pin it here too so
        # a site that regressed to a message-only construction fails on this
        # assertion rather than on a type-check that runs elsewhere. storage_key
        # is keyword-only, so it can only appear as a keyword.
        assert any(kw.arg == "storage_key" for kw in node.keywords), (
            f"managed_file_ref.py:{node.lineno} constructs "
            f"{callee_name} without storage_key=; the key must ride the "
            "attribute, not just be absent from the message"
        )
        if node.args:
            message = node.args[0]
        else:
            message = next(
                (kw.value for kw in node.keywords if kw.arg == "message"), None
            )
        if message is None:
            continue
        checked += 1
        is_literal = isinstance(message, ast.Constant) and isinstance(
            message.value, str
        )
        assert is_literal or isinstance(message, ast.Name), (
            f"managed_file_ref.py:{node.lineno} builds the message dynamically "
            f"({ast.unparse(message)!r}); it must be a string literal or a "
            "module constant -- the identifier belongs in storage_key= so "
            "str(exc) stays safe"
        )
    # The count is part of the contract: consolidating constructions into a
    # helper would leave only the helper's own body visible to this walk while
    # every real call site goes dark, and an aliased construction is invisible
    # to the name match -- both would silently shrink coverage. A changed count
    # means a construction was added, removed, or hidden: update it
    # deliberately and give the new site its coverage story.
    assert checked == 8, (
        f"expected 8 constructions (7 wraps + the integrity raise), found "
        f"{checked} -- a construction was added, removed, aliased, or moved "
        "behind a helper; update this count deliberately"
    )
