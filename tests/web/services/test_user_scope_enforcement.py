"""End-to-end enforcement of user-scoped storage handles at the call sites.

ManagedFileRef defaults to a handle scoped to ``users/{record.user_id}``;
these tests prove a record whose storage_key targets another user's prefix
cannot read, write, sign, adopt, or delete through any entry point.
"""

from pathlib import Path

import pytest

from xagent.core.execution_scope import (
    ExecutionScope,
    reset_execution_scope,
    set_execution_scope,
    set_execution_scope_resolver,
)
from xagent.core.file_storage import StorageKeyScopeError
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.managed_file_ref import (
    DurableStorageOperationError,
    ManagedFileRef,
)

_ISOLATED_SCOPE = ExecutionScope(
    workspace_segments=("clients", "3", "end_users", "7"),
    isolate_external_dirs=True,
)
# Same segments, but not isolated: the handle must stay at the owner root so
# legitimate shared owner-level reads still work (mirrors the sandbox
# filesystem allowlist, which only narrows under ``isolate_external_dirs``).
_NON_ISOLATED_SCOPE = ExecutionScope(
    workspace_segments=("clients", "3", "end_users", "7"),
    isolate_external_dirs=False,
)


@pytest.fixture
def storage_env(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized"))
    get_unscoped_file_storage.cache_clear()
    yield tmp_path
    get_unscoped_file_storage.cache_clear()


@pytest.fixture
def clear_scope_resolver():
    """Start and end each test with no registered resolver."""
    set_execution_scope_resolver(None)
    yield
    set_execution_scope_resolver(None)


def _record(local_path: Path, **overrides) -> UploadedFile:
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


def _foreign_key_record(local_path: Path) -> UploadedFile:
    return _record(
        local_path,
        storage_key="users/8/uploads/file-123/source.txt",
        storage_backend="file",
        storage_status="available",
    )


def test_round_trip_through_default_user_scoped_storage(storage_env, tmp_path):
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir()
    source.write_text("scoped round trip", encoding="utf-8")
    record = _record(source)

    stored = ManagedFileRef(record).sync_to_durable()
    assert stored.key == "users/7/uploads/file-123/source.txt"
    assert record.storage_status == "available"

    source.unlink()
    restored = ManagedFileRef(record).ensure_local()
    assert restored.read_text(encoding="utf-8") == "scoped round trip"

    ManagedFileRef(record).delete_durable()
    assert not get_unscoped_file_storage().exists(stored.key)


def test_sync_to_durable_rejects_foreign_explicit_key(storage_env, tmp_path):
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir()
    source.write_text("data", encoding="utf-8")
    record = _record(source)

    with pytest.raises(DurableStorageOperationError) as excinfo:
        ManagedFileRef(record).sync_to_durable(
            storage_key="users/8/uploads/file-123/source.txt"
        )
    assert isinstance(excinfo.value.__cause__, StorageKeyScopeError)
    assert not get_unscoped_file_storage().exists("users/8/uploads/file-123/source.txt")


def test_restore_rejects_foreign_storage_key(storage_env, tmp_path):
    record = _foreign_key_record(tmp_path / "uploads" / "missing.txt")

    with pytest.raises(DurableStorageOperationError) as excinfo:
        ManagedFileRef(record).ensure_local()
    assert isinstance(excinfo.value.__cause__, StorageKeyScopeError)

    with pytest.raises(DurableStorageOperationError) as excinfo:
        ManagedFileRef(record).materialize()
    assert isinstance(excinfo.value.__cause__, StorageKeyScopeError)


def test_signed_url_never_issued_for_foreign_storage_key(storage_env, tmp_path):
    # signed_access_url degrades to None when the checksum probe fails; the
    # scope check makes that probe fail for foreign keys, so no URL is issued
    # even though the foreign object exists and the checksum matches.
    foreign = get_unscoped_file_storage().put_bytes(
        b"foreign", "users/8/uploads/file-123/missing.txt"
    )
    record = _record(
        tmp_path / "uploads" / "missing.txt",
        storage_key=foreign.key,
        storage_backend="file",
        storage_status="available",
        checksum=foreign.checksum,
    )

    assert ManagedFileRef(record).signed_access_url(expires=300) is None


def test_delete_durable_rejects_foreign_storage_key(storage_env, tmp_path):
    record = _foreign_key_record(tmp_path / "uploads" / "missing.txt")

    with pytest.raises(StorageKeyScopeError):
        ManagedFileRef(record).delete_durable()


def test_adopt_existing_object_rejects_foreign_expected_key(storage_env, tmp_path):
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir()
    source.write_text("data", encoding="utf-8")
    record = _record(source)
    # The foreign object exists, so a scope bypass would return "adopted".
    get_unscoped_file_storage().put_bytes(
        b"foreign", "users/8/uploads/file-123/source.txt"
    )

    with pytest.raises(DurableStorageOperationError) as excinfo:
        ManagedFileRef(record).adopt_existing_object(
            "users/8/uploads/file-123/source.txt"
        )
    assert isinstance(excinfo.value.__cause__, StorageKeyScopeError)


def test_separator_aware_scope_for_record_owner(storage_env, tmp_path):
    # user 1's handle must not admit a users/10 key.
    record = _record(
        tmp_path / "uploads" / "missing.txt",
        user_id=1,
        storage_key="users/10/uploads/file-123/source.txt",
        storage_backend="file",
        storage_status="available",
    )

    with pytest.raises(StorageKeyScopeError):
        ManagedFileRef(record).delete_durable()


def test_record_without_owner_cannot_bind_default_scope(storage_env, tmp_path):
    record = _record(tmp_path / "uploads" / "missing.txt", user_id=None)

    with pytest.raises(ValueError, match="user_id is required"):
        ManagedFileRef(record)


# --- scope-aware handle binding (#828 durable-storage half) -----------------


def test_unscoped_construction_binds_owner_root(storage_env, tmp_path):
    record = _record(tmp_path / "uploads" / "missing.txt")
    assert ManagedFileRef(record).storage.prefix == "users/7"


def test_explicit_isolated_scope_narrows_handle_prefix(storage_env, tmp_path):
    record = _record(tmp_path / "uploads" / "missing.txt")
    ref = ManagedFileRef(record, execution_scope=_ISOLATED_SCOPE)
    assert ref.storage.prefix == "users/7/clients/3/end_users/7"


def test_non_isolated_scope_keeps_owner_root(storage_env, tmp_path):
    # A scope with segments but no isolation must NOT narrow the handle, or it
    # would block the shared owner-level reads such executions rely on.
    record = _record(tmp_path / "uploads" / "missing.txt")
    ref = ManagedFileRef(record, execution_scope=_NON_ISOLATED_SCOPE)
    assert ref.storage.prefix == "users/7"


def test_ambient_isolated_scope_narrows_handle_prefix(storage_env, tmp_path):
    record = _record(tmp_path / "uploads" / "missing.txt")
    token = set_execution_scope(_ISOLATED_SCOPE)
    try:
        assert ManagedFileRef(record).storage.prefix == "users/7/clients/3/end_users/7"
    finally:
        reset_execution_scope(token)


def test_explicit_scope_overrides_ambient(storage_env, tmp_path):
    record = _record(tmp_path / "uploads" / "missing.txt")
    token = set_execution_scope(_NON_ISOLATED_SCOPE)
    try:
        ref = ManagedFileRef(record, execution_scope=_ISOLATED_SCOPE)
        assert ref.storage.prefix == "users/7/clients/3/end_users/7"
    finally:
        reset_execution_scope(token)


def test_explicit_unscoped_overrides_ambient(storage_env, tmp_path):
    """Explicit None is an owner-root decision, not an omitted argument."""

    record = _record(tmp_path / "uploads" / "missing.txt")
    token = set_execution_scope(_ISOLATED_SCOPE)
    try:
        ref = ManagedFileRef(record, execution_scope=None)
        assert ref.storage.prefix == "users/7"
    finally:
        reset_execution_scope(token)


def test_sync_to_durable_writes_into_scoped_subtree(storage_env, tmp_path):
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir()
    source.write_text("scoped upload", encoding="utf-8")
    record = _record(source)

    stored = ManagedFileRef(record, execution_scope=_ISOLATED_SCOPE).sync_to_durable()
    assert stored.key == "users/7/clients/3/end_users/7/uploads/file-123/source.txt"
    assert record.storage_status == "available"

    # The same isolated handle round-trips its own object back.
    source.unlink()
    restored = ManagedFileRef(record, execution_scope=_ISOLATED_SCOPE).ensure_local()
    assert restored.read_text(encoding="utf-8") == "scoped upload"


def test_isolated_handle_rejects_sibling_end_user_key(storage_env, tmp_path):
    # Same owner, sibling end user (end_users/8). Without the handle narrowing
    # this key sits under ``users/7`` and would be reachable; the scoped handle
    # must reject it — the defense-in-depth this issue is about.
    record = _record(
        tmp_path / "uploads" / "missing.txt",
        storage_key="users/7/clients/3/end_users/8/uploads/file-123/source.txt",
        storage_backend="file",
        storage_status="available",
    )

    with pytest.raises(StorageKeyScopeError):
        ManagedFileRef(record, execution_scope=_ISOLATED_SCOPE).delete_durable()


# --- resolver/snapshot fallback when the turn contextvar is absent ----------
# Off-turn paths (bot/builder-chat handlers) never enter turn_execution_scope,
# so the ambient contextvar is None while a workforce sub-task's scope is still
# recoverable from the per-task resolver/snapshot keyed on record.task_id.


def test_resolver_narrows_handle_when_contextvar_absent(
    storage_env, tmp_path, clear_scope_resolver
):
    set_execution_scope_resolver(
        lambda task_id: _ISOLATED_SCOPE if task_id == "99" else None
    )
    record = _record(tmp_path / "uploads" / "missing.txt", task_id=99)
    # No ambient contextvar, no explicit scope: fall back to the resolver.
    assert ManagedFileRef(record).storage.prefix == "users/7/clients/3/end_users/7"


def test_contextvar_beats_resolver(storage_env, tmp_path, clear_scope_resolver):
    set_execution_scope_resolver(lambda task_id: _NON_ISOLATED_SCOPE)
    record = _record(tmp_path / "uploads" / "missing.txt", task_id=99)
    token = set_execution_scope(_ISOLATED_SCOPE)
    try:
        assert ManagedFileRef(record).storage.prefix == "users/7/clients/3/end_users/7"
    finally:
        reset_execution_scope(token)


def test_explicit_scope_beats_resolver(storage_env, tmp_path, clear_scope_resolver):
    set_execution_scope_resolver(lambda task_id: _ISOLATED_SCOPE)
    record = _record(tmp_path / "uploads" / "missing.txt", task_id=99)
    ref = ManagedFileRef(record, execution_scope=_NON_ISOLATED_SCOPE)
    assert ref.storage.prefix == "users/7"


def test_resolver_not_consulted_without_task_id(
    storage_env, tmp_path, clear_scope_resolver
):
    def _boom(task_id):
        raise AssertionError("resolver must not run when the record has no task_id")

    set_execution_scope_resolver(_boom)
    record = _record(tmp_path / "uploads" / "missing.txt")  # task_id defaults to None
    assert ManagedFileRef(record).storage.prefix == "users/7"


def test_sync_to_durable_uses_resolved_scope(
    storage_env, tmp_path, clear_scope_resolver
):
    set_execution_scope_resolver(
        lambda task_id: _ISOLATED_SCOPE if task_id == "99" else None
    )
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir()
    source.write_text("workforce upload", encoding="utf-8")
    record = _record(source, task_id=99)

    stored = ManagedFileRef(record).sync_to_durable()
    assert stored.key == "users/7/clients/3/end_users/7/uploads/file-123/source.txt"
