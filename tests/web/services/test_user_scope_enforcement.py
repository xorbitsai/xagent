"""End-to-end enforcement of user-scoped storage handles at the call sites.

ManagedFileRef defaults to a handle scoped to ``users/{record.user_id}``;
these tests prove a record whose storage_key targets another user's prefix
cannot read, write, sign, adopt, or delete through any entry point.
"""

from pathlib import Path

import pytest

from tests.shared.execution_scope import register_scope_resolver
from xagent.core.execution_scope import (
    DeferToSnapshot,
    ExecutionScope,
    ExecutionScopeAuthorityError,
    ExecutionScopeResolverContractError,
    reset_execution_scope,
    set_execution_scope,
    set_execution_scope_snapshot_loader,
)
from xagent.core.file_storage import StorageKeyScopeError
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services.managed_file_ref import ManagedFileRef

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

    with pytest.raises(StorageKeyScopeError):
        ManagedFileRef(record).sync_to_durable(
            storage_key="users/8/uploads/file-123/source.txt"
        )
    assert not get_unscoped_file_storage().exists("users/8/uploads/file-123/source.txt")


def test_restore_rejects_foreign_storage_key(storage_env, tmp_path):
    record = _foreign_key_record(tmp_path / "uploads" / "missing.txt")

    with pytest.raises(StorageKeyScopeError):
        ManagedFileRef(record).ensure_local()

    with pytest.raises(StorageKeyScopeError):
        ManagedFileRef(record).materialize()


def test_signed_url_never_issued_for_foreign_storage_key(storage_env, tmp_path):
    # No URL may be issued for a foreign key even though the foreign object
    # exists and its checksum matches. The containment violation is a
    # permanent authority fault, so it propagates to be classified once at the
    # application boundary rather than being reported as an unavailable
    # checksum -- which would read as a transient reason to fall back to
    # backend-mediated access, a fallback that hits the same violation anyway.
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

    with pytest.raises(StorageKeyScopeError):
        ManagedFileRef(record).signed_access_url(expires=300)


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

    with pytest.raises(StorageKeyScopeError):
        ManagedFileRef(record).adopt_existing_object(
            "users/8/uploads/file-123/source.txt"
        )


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
# Off-turn paths (bot/builder-chat handlers) never enter an ExecutionScopeContext,
# so the ambient contextvar is None while a workforce sub-task's scope is still
# recoverable from the per-task resolver/snapshot keyed on record.task_id.


def test_resolver_narrows_handle_when_contextvar_absent(storage_env, tmp_path):
    register_scope_resolver(
        lambda task_id: _ISOLATED_SCOPE if task_id == "99" else None,
    )
    record = _record(tmp_path / "uploads" / "missing.txt", task_id=99)
    # No ambient contextvar, no explicit scope: fall back to the resolver.
    # The record carries no durable key, so the per-task recovery is owed
    # rather than already done -- it settles when an operation first needs a
    # namespace, and the handle it settles on is the resolver's.
    ref = ManagedFileRef(record)
    assert ref.storage is None
    assert ref._bound_storage().prefix == "users/7/clients/3/end_users/7"


def test_contextvar_beats_resolver(storage_env, tmp_path):
    register_scope_resolver(
        lambda task_id: _NON_ISOLATED_SCOPE,
    )
    record = _record(tmp_path / "uploads" / "missing.txt", task_id=99)
    token = set_execution_scope(_ISOLATED_SCOPE)
    try:
        assert ManagedFileRef(record).storage.prefix == "users/7/clients/3/end_users/7"
    finally:
        reset_execution_scope(token)


def test_explicit_scope_beats_resolver(storage_env, tmp_path):
    register_scope_resolver(lambda task_id: _ISOLATED_SCOPE)
    record = _record(tmp_path / "uploads" / "missing.txt", task_id=99)
    ref = ManagedFileRef(record, execution_scope=_NON_ISOLATED_SCOPE)
    assert ref.storage.prefix == "users/7"


def test_resolver_not_consulted_without_task_id(storage_env, tmp_path):
    def _boom(task_id):
        raise AssertionError("resolver must not run when the record has no task_id")

    register_scope_resolver(_boom)
    record = _record(tmp_path / "uploads" / "missing.txt")  # task_id defaults to None
    assert ManagedFileRef(record).storage.prefix == "users/7"


def test_sync_to_durable_uses_resolved_scope(storage_env, tmp_path):
    register_scope_resolver(
        lambda task_id: _ISOLATED_SCOPE if task_id == "99" else None,
    )
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir()
    source.write_text("workforce upload", encoding="utf-8")
    record = _record(source, task_id=99)

    stored = ManagedFileRef(record).sync_to_durable()
    assert stored.key == "users/7/clients/3/end_users/7/uploads/file-123/source.txt"


# --- read path skips off-turn re-resolution (#296) ------
# A record that already has a durable storage_key fixes its own location;
# off-turn re-resolving the scope is unnecessary and, on a resolver/snapshot
# drift, would make an already-legitimate key look foreign to a narrower
# re-derived scope. Only a record with no storage_key yet (a new object about
# to be written) still resolves off-turn.


def test_existing_storage_key_skips_off_turn_resolution(storage_env, tmp_path):
    def _boom(task_id):
        raise AssertionError(
            "off-turn resolution must not run when the record already has "
            "a durable storage_key"
        )

    register_scope_resolver(_boom)
    record = _record(
        tmp_path / "uploads" / "missing.txt",
        task_id=99,
        storage_key="users/7/clients/3/end_users/7/uploads/file-123/source.txt",
        storage_backend="file",
        storage_status="available",
    )

    # Owner root, not the resolver's (never-called) narrower prefix.
    assert ManagedFileRef(record).storage.prefix == "users/7"


def test_existing_storage_key_binding_tolerates_resolver_snapshot_mismatch(
    storage_env, tmp_path
):
    """An off-turn authority mismatch would otherwise fail closed (see
    ``resolve_execution_scope_off_turn``); the read path never even reaches
    that check because it does not resolve off-turn at all."""
    register_scope_resolver(lambda task_id: _ISOLATED_SCOPE)
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(workspace_segments=("other-tenant",))
    )
    record = _record(
        tmp_path / "uploads" / "missing.txt",
        task_id=99,
        storage_key="users/7/uploads/file-123/source.txt",
        storage_backend="file",
        storage_status="available",
    )

    assert ManagedFileRef(record).storage.prefix == "users/7"


def test_sync_to_durable_fails_closed_on_mismatch_for_recovered_scope(
    storage_env, tmp_path
):
    """No storage_key yet (a fresh upload): construction always recovers the
    scope off-turn (see ``__post_init__``) and never raises here, since a
    keyless ref might just as well be serving a read. Choosing the namespace
    new bytes land under is the actual authority decision, so
    ``sync_to_durable`` re-resolves fail-closed right before composing the
    key -- getting it wrong would silently and durably place the object
    under the wrong tenant's subtree. No object may be written under either
    candidate prefix."""
    register_scope_resolver(lambda task_id: _ISOLATED_SCOPE)
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(workspace_segments=("other-tenant",))
    )
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir()
    source.write_text("data", encoding="utf-8")
    record = _record(source, task_id=99, storage_status="pending")

    ref = ManagedFileRef(record)  # construction downgrades, does not raise

    with pytest.raises(ExecutionScopeAuthorityError):
        ref.sync_to_durable()

    assert record.storage_status == "pending"
    objects_root = tmp_path / "objects" / "users" / "7"
    assert not (objects_root / "clients").exists()
    assert not (objects_root / "other-tenant").exists()


def test_read_path_downgrades_on_mismatch_for_pending_record(storage_env, tmp_path):
    """No storage_key yet is also the normal state of a record read while its
    upload is still pending (see build_uploaded_file_record). A read has
    local storage to fall back to, so construction downgrades a
    resolver/snapshot mismatch to the resolver's answer and still serves the
    file, instead of raising ExecutionScopeAuthorityError into an unhandled
    500 (the namespace decision that must fail closed is deferred to
    ``sync_to_durable``, see the test above)."""
    register_scope_resolver(lambda task_id: _ISOLATED_SCOPE)
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(workspace_segments=("other-tenant",))
    )
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir()
    source.write_text("pending upload", encoding="utf-8")
    record = _record(source, task_id=99, storage_status="pending")

    # Serving the local copy needs no namespace at all, so nothing is
    # resolved and nothing can be disputed.
    ref = ManagedFileRef(record)
    assert ref.ensure_local().read_text(encoding="utf-8") == "pending upload"
    assert ref.storage is None


def test_materialize_downgrades_on_mismatch_for_pending_record(storage_env, tmp_path):
    """``materialize()`` (see ``Workspace``'s
    ``ManagedFileRef(record).materialize()`` call, and similarly ``kb.py``'s
    ``ensure_local``/``delete_durable`` calls) reads or addresses bytes
    already placed, never a namespace decision, so it must not be blocked by
    a resolver/snapshot mismatch that construction now downgrades instead of
    raising."""
    register_scope_resolver(lambda task_id: _ISOLATED_SCOPE)
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(workspace_segments=("other-tenant",))
    )
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir()
    source.write_text("pending upload", encoding="utf-8")
    record = _record(source, task_id=99, storage_status="pending")

    ref = ManagedFileRef(record)
    assert ref.materialize().read_text(encoding="utf-8") == "pending upload"
    assert ref.storage is None


def _pending_record(tmp_path: Path) -> UploadedFile:
    source = tmp_path / "uploads" / "source.txt"
    source.parent.mkdir(exist_ok=True)
    source.write_text("pending upload", encoding="utf-8")
    return _record(source, task_id=99, storage_status="pending")


def test_read_serves_when_an_abstaining_resolver_disagrees_with_the_snapshot(
    storage_env, tmp_path
):
    """An abstention mismatch has no authoritative value to downgrade to, so
    it fails closed wherever a namespace is required. A read of a pending
    record requires none, and must still be served."""
    register_scope_resolver(
        lambda task_id: DeferToSnapshot(fallback=ExecutionScope()),
    )
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(workspace_segments=("wider",))
    )
    ref = ManagedFileRef(_pending_record(tmp_path))

    assert ref.ensure_local().read_text(encoding="utf-8") == "pending upload"
    assert ref.materialize().read_text(encoding="utf-8") == "pending upload"


def test_read_serves_when_the_resolver_breaks_its_return_contract(
    storage_env, tmp_path
):
    register_scope_resolver(lambda task_id: "not-a-scope")
    ref = ManagedFileRef(_pending_record(tmp_path))

    assert ref.ensure_local().read_text(encoding="utf-8") == "pending upload"


def test_read_serves_when_the_snapshot_loader_raises(storage_env, tmp_path):
    """No resolver registered is this repository's shape today, and there the
    loader's answer is the whole authority -- a database hiccup while reading
    it must not take a read-only endpoint down with it."""

    def _explode(task_id):
        raise RuntimeError("snapshot row unreadable")

    set_execution_scope_snapshot_loader(_explode)
    ref = ManagedFileRef(_pending_record(tmp_path))

    assert ref.ensure_local().read_text(encoding="utf-8") == "pending upload"


def test_sync_to_durable_still_fails_closed_when_the_resolver_abstains_and_disagrees(
    storage_env, tmp_path
):
    """The half that must not regress: deferring the resolution must not
    weaken the check that runs where a namespace is chosen for new bytes."""
    register_scope_resolver(
        lambda task_id: DeferToSnapshot(fallback=ExecutionScope()),
    )
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(workspace_segments=("wider",))
    )
    record = _pending_record(tmp_path)
    ref = ManagedFileRef(record)

    with pytest.raises(ExecutionScopeAuthorityError):
        ref.sync_to_durable()

    assert record.storage_status == "pending"
    objects_root = tmp_path / "objects" / "users" / "7"
    assert not (objects_root / "wider").exists()
    assert not (objects_root / "uploads").exists()


def test_sync_to_durable_still_fails_closed_when_the_resolver_breaks_its_contract(
    storage_env, tmp_path
):
    register_scope_resolver(lambda task_id: "not-a-scope")
    record = _pending_record(tmp_path)
    ref = ManagedFileRef(record)

    with pytest.raises(ExecutionScopeResolverContractError):
        ref.sync_to_durable()

    assert record.storage_status == "pending"
    assert not (tmp_path / "objects" / "users" / "7" / "uploads").exists()
