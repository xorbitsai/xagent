"""Tests for DynamicMemoryStoreManager embedding-config change detection."""

from types import SimpleNamespace
from typing import Any

from xagent.web.dynamic_memory_store import DynamicMemoryStoreManager


class FakeLanceStore:
    def __init__(self, model: Any) -> None:
        self.model = model


def _manager_with_fake_db(monkeypatch, model_holder: dict) -> DynamicMemoryStoreManager:
    manager = DynamicMemoryStoreManager()
    monkeypatch.setattr(
        manager, "_get_embedding_model_from_db", lambda: model_holder["model"]
    )
    monkeypatch.setattr(
        manager,
        "_create_lancedb_store",
        lambda model: FakeLanceStore(model),
    )
    return manager


def _model(model_id: int, updated_at: str, api_key: str) -> Any:
    return SimpleNamespace(
        id=model_id,
        updated_at=updated_at,
        api_key=api_key,
        model_provider="dashscope",
        dimension=1024,
    )


def test_key_rotation_on_same_model_rebuilds_store(monkeypatch) -> None:
    holder = {"model": _model(2, "2026-07-17 10:00:00", "old-key")}
    manager = _manager_with_fake_db(monkeypatch, holder)

    first = manager.get_memory_store()
    assert isinstance(first, FakeLanceStore)
    assert first.model.api_key == "old-key"

    # Same model id, but the row was edited (key rotation bumps updated_at).
    holder["model"] = _model(2, "2026-07-17 11:00:00", "new-key")
    assert manager.check_embedding_model_change() is True
    second = manager.get_memory_store()
    assert isinstance(second, FakeLanceStore)
    assert second.model.api_key == "new-key"
    assert second is not first


def test_unchanged_model_keeps_store_instance(monkeypatch) -> None:
    holder = {"model": _model(2, "2026-07-17 10:00:00", "key")}
    manager = _manager_with_fake_db(monkeypatch, holder)

    first = manager.get_memory_store()
    assert manager.check_embedding_model_change() is False
    assert manager.get_memory_store() is first
