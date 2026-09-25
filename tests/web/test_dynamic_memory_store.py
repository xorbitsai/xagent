"""DynamicMemoryStoreManager: what a configuration change must *not* do.

Before the lifecycle redesign this manager rebuilt its store online whenever the
embedding configuration moved. It no longer does: the runtime consumes the
explicit global authority, and every change to it takes effect through
quiescence and an all-worker restart. These tests pin the inversion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pytest
from pydantic import SecretStr

from xagent.web import dynamic_memory_store as manager_module
from xagent.web.dynamic_memory_store import DynamicMemoryStoreManager
from xagent.web.memory_lifecycle import (
    AdmissionResult,
    MemoryLifecycleState,
    MemoryLifecycleStatus,
    MemoryUnavailableError,
)
from xagent.web.revocable_memory_store import unwrap_memory_store
from xagent.web.services.global_memory_embedding_authority import (
    CREDENTIAL_CONFIGURED,
    CredentialSource,
    GlobalMemoryEmbeddingAuthoritySnapshot,
)


class FakeLanceStore:
    def __init__(self, snapshot: Any) -> None:
        self.snapshot = snapshot


def _snapshot(
    *, model_name: str = "text-embedding-3-small", api_key: str = "key"
) -> GlobalMemoryEmbeddingAuthoritySnapshot:
    now = datetime.now(timezone.utc)
    return GlobalMemoryEmbeddingAuthoritySnapshot(
        provider="openai",
        model_name=model_name,
        endpoint="https://api.openai.com/v1/embeddings",
        dimension=1024,
        instruct=None,
        max_retries=3,
        credential_source=CredentialSource.ORGANIZATION_OWNED,
        global_sharing_consent=True,
        consented_by_actor_subject="admin",
        consented_at=now,
        created_at=now,
        updated_at=now,
        credential_status=CREDENTIAL_CONFIGURED,
        api_key=SecretStr(api_key),
        credential_identity=f"identity-for-{api_key}",
    )


def _manager_with_fake_authority(
    monkeypatch, holder: dict
) -> DynamicMemoryStoreManager:
    """A manager whose authority read and admission are both fakes."""

    def read() -> Optional[GlobalMemoryEmbeddingAuthoritySnapshot]:
        return holder["snapshot"]

    def admit(snapshot, **_kwargs) -> AdmissionResult:
        return AdmissionResult(
            MemoryLifecycleStatus(MemoryLifecycleState.READY),
            store=FakeLanceStore(snapshot),
            vector_space_fingerprint=snapshot.vector_space_fingerprint(),
        )

    monkeypatch.setattr(manager_module, "_read_authority_snapshot", read)
    monkeypatch.setattr(manager_module, "admit_authority_storage", admit)
    manager = DynamicMemoryStoreManager()
    # Admission is startup-only; nothing publishes a store lazily any more.
    manager.admit()
    return manager


def test_key_rotation_on_the_same_authority_does_not_rebuild(monkeypatch) -> None:
    """Rotating the credential is not a vector-space change, and never reloads."""
    holder = {"snapshot": _snapshot(api_key="old-key")}
    manager = _manager_with_fake_authority(monkeypatch, holder)

    first = manager.get_memory_store()
    assert isinstance(unwrap_memory_store(first), FakeLanceStore)
    assert unwrap_memory_store(first).snapshot.api_key.get_secret_value() == "old-key"

    holder["snapshot"] = _snapshot(api_key="new-key")

    assert manager.check_embedding_model_change() is False
    second = manager.get_memory_store()
    assert second is first
    assert unwrap_memory_store(second).snapshot.api_key.get_secret_value() == "old-key"
    # The reference handed out before the rotation is still usable.
    assert first.revoked is False


def test_changed_vector_space_revokes_instead_of_rebuilding(monkeypatch) -> None:
    holder = {"snapshot": _snapshot()}
    manager = _manager_with_fake_authority(monkeypatch, holder)
    assert isinstance(unwrap_memory_store(manager.get_memory_store()), FakeLanceStore)

    holder["snapshot"] = _snapshot(model_name="text-embedding-3-large")

    assert manager.check_embedding_model_change() is True
    with pytest.raises(MemoryUnavailableError) as raised:
        manager.get_memory_store()
    assert raised.value.status.state is MemoryLifecycleState.RESTART_REQUIRED


def test_unchanged_authority_keeps_store_instance(monkeypatch) -> None:
    holder = {"snapshot": _snapshot()}
    manager = _manager_with_fake_authority(monkeypatch, holder)

    first = manager.get_memory_store()
    assert manager.check_embedding_model_change() is False
    assert manager.get_memory_store() is first


def test_authority_read_happens_under_the_lock(monkeypatch) -> None:
    holder = {"snapshot": _snapshot()}
    manager = _manager_with_fake_authority(monkeypatch, holder)

    def read_under_lock() -> GlobalMemoryEmbeddingAuthoritySnapshot:
        assert manager._lock._is_owned()  # type: ignore[attr-defined]
        return holder["snapshot"]

    monkeypatch.setattr(manager_module, "_read_authority_snapshot", read_under_lock)

    assert isinstance(unwrap_memory_store(manager.get_memory_store()), FakeLanceStore)


# --------------------------------------------------------------------------
# A store that was already handed out is revoked too.
#
# Revoking the manager's publication does not reach what callers already hold:
# AgentService keeps its store in self.memory, and the agent cache's hit path
# re-checks owner and scope only, never memory policy. The published store is
# therefore a generation-bound proxy, so a cached agent -- and an execution
# already in flight -- fails closed the moment the publication is revoked.
# --------------------------------------------------------------------------


class RecordingStore:
    """A base store that records every call that actually reached it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def add(self, note):
        self.calls.append("add")
        return "added"

    def get(self, note_id):
        self.calls.append("get")
        return "got"

    def update(self, note):
        self.calls.append("update")
        return "updated"

    def delete(self, note_id):
        self.calls.append("delete")
        return "deleted"

    def search(self, query, k=5, filters=None, similarity_threshold=None):
        self.calls.append("search")
        return []

    def clear(self):
        self.calls.append("clear")

    def list_all(self, filters=None):
        self.calls.append("list_all")
        return []

    def get_stats(self):
        self.calls.append("get_stats")
        return {}

    def delete_by_scope_dimension(self, dim_key, value):
        self.calls.append("delete_by_scope_dimension")
        return "deleted"

    def list_scope_dimension_values(self, dim_key):
        self.calls.append("list_scope_dimension_values")
        return set()


def _manager_with_recording_store(monkeypatch, holder: dict):
    """A manager publishing a RecordingStore, so reach-through is observable."""
    base = RecordingStore()

    def read():
        return holder["snapshot"]

    def admit(snapshot, **_kwargs) -> AdmissionResult:
        return AdmissionResult(
            MemoryLifecycleStatus(MemoryLifecycleState.READY),
            store=base,
            vector_space_fingerprint=snapshot.vector_space_fingerprint(),
        )

    monkeypatch.setattr(manager_module, "_read_authority_snapshot", read)
    monkeypatch.setattr(manager_module, "admit_authority_storage", admit)
    manager = DynamicMemoryStoreManager()
    manager.admit()
    return manager, base


def test_handed_out_reference_is_revoked_by_vector_space_drift(monkeypatch) -> None:
    """The caller never asks the manager again; it must still fail closed."""
    holder = {"snapshot": _snapshot()}
    manager, base = _manager_with_recording_store(monkeypatch, holder)

    # A caller (AgentService.memory, a memory tool) takes its reference once.
    cached = manager.get_memory_store()
    assert cached.get("note-1") == "got"
    assert base.calls == ["get"]

    holder["snapshot"] = _snapshot(model_name="text-embedding-3-large")
    # The manager notices the drift on its own drift-check path.
    assert manager.check_embedding_model_change() is True

    # Read fails closed...
    with pytest.raises(MemoryUnavailableError) as read_error:
        cached.get("note-1")
    assert read_error.value.status.state is MemoryLifecycleState.RESTART_REQUIRED

    # ...and so does write.
    with pytest.raises(MemoryUnavailableError) as write_error:
        cached.add(object())
    assert write_error.value.status.state is MemoryLifecycleState.RESTART_REQUIRED

    # Nothing reached the old-space store after revocation.
    assert base.calls == ["get"]
    assert cached.revoked is True


def test_drift_noticed_by_a_new_acquire_revokes_the_old_reference(monkeypatch) -> None:
    """The next task starting is enough to fence the previous one's store."""
    holder = {"snapshot": _snapshot()}
    manager, base = _manager_with_recording_store(monkeypatch, holder)
    cached = manager.get_memory_store()
    assert cached.list_all() == []

    holder["snapshot"] = _snapshot(model_name="text-embedding-3-large")
    # A later task resolves its memory policy and is refused a store. That
    # same acquire is what revokes the reference the earlier task still holds.
    with pytest.raises(MemoryUnavailableError):
        manager.get_memory_store()

    with pytest.raises(MemoryUnavailableError):
        cached.list_all()
    assert base.calls == ["list_all"]


@pytest.mark.parametrize(
    "operation",
    [
        lambda store: store.add(object()),
        lambda store: store.get("note-1"),
        lambda store: store.update(object()),
        lambda store: store.delete("note-1"),
        lambda store: store.search("anything"),
        lambda store: store.clear(),
        lambda store: store.list_all(),
        lambda store: store.get_stats(),
        lambda store: store.delete_by_scope_dimension("tenant", "acme"),
        lambda store: store.list_scope_dimension_values("tenant"),
    ],
)
def test_every_memory_operation_fails_closed_after_revocation(
    monkeypatch, operation
) -> None:
    """No method may be a hole through the revocation check."""
    holder = {"snapshot": _snapshot()}
    manager, base = _manager_with_recording_store(monkeypatch, holder)
    cached = manager.get_memory_store()

    holder["snapshot"] = _snapshot(model_name="text-embedding-3-large")
    assert manager.check_embedding_model_change() is True

    with pytest.raises(MemoryUnavailableError):
        operation(cached)
    assert base.calls == []


def test_credential_rotation_does_not_revoke_a_handed_out_reference(
    monkeypatch,
) -> None:
    holder = {"snapshot": _snapshot(api_key="old-key")}
    manager, base = _manager_with_recording_store(monkeypatch, holder)
    cached = manager.get_memory_store()

    holder["snapshot"] = _snapshot(api_key="new-key")
    assert manager.check_embedding_model_change() is False

    assert cached.get("note-1") == "got"
    assert cached.add(object()) == "added"
    assert cached.revoked is False
    assert base.calls == ["get", "add"]


def test_description_only_authority_edit_does_not_revoke(monkeypatch) -> None:
    """A retry-budget change carries no vector-space meaning."""
    holder = {"snapshot": _snapshot()}
    manager, base = _manager_with_recording_store(monkeypatch, holder)
    cached = manager.get_memory_store()

    replacement = _snapshot()
    object.__setattr__(replacement, "max_retries", 9)
    # Guard the guard: a silently-ignored mutation would make this vacuous.
    assert replacement.max_retries == 9
    assert replacement.max_retries != _snapshot().max_retries
    holder["snapshot"] = replacement

    assert manager.check_embedding_model_change() is False
    assert cached.get("note-1") == "got"
    assert cached.revoked is False
    assert base.calls == ["get"]


def test_transient_database_failure_does_not_revoke(monkeypatch) -> None:
    """A database hiccup is not an identity change and must not fence memory."""
    holder = {"snapshot": _snapshot()}
    manager, base = _manager_with_recording_store(monkeypatch, holder)
    cached = manager.get_memory_store()

    def failing_read():
        raise manager_module.AuthorityUnreadable("database down")

    monkeypatch.setattr(manager_module, "_read_authority_snapshot", failing_read)

    assert manager.check_embedding_model_change() is False
    assert cached.get("note-1") == "got"
    assert cached.add(object()) == "added"
    assert cached.revoked is False
    assert base.calls == ["get", "add"]


def test_store_info_reports_the_adapter_through_the_proxy(monkeypatch) -> None:
    """The revocation wrapper must not become the reported store type."""
    holder = {"snapshot": _snapshot()}
    manager, _base = _manager_with_recording_store(monkeypatch, holder)

    info = manager.get_store_info()

    assert info["store_type"] == "RecordingStore"
    assert info["state"] == MemoryLifecycleState.READY.value
