"""``create_memory_store`` must not open a database engine of its own.

It used to build a second SQLAlchemy engine so it could scan the model hub for
any embedding model at all. That scan is gone -- the runtime consumes only the
explicit global memory embedding authority, read through the shared web session
-- and so is the engine whose bound parameters could otherwise have surfaced
the encrypted API key column on a bind or commit failure. This test keeps that
engine from coming back, and keeps the forbidden model-hub scan from coming
back with it.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import sqlalchemy

from xagent.web import dynamic_memory_store, memory_utils
from xagent.web.memory_lifecycle import MemoryUnavailableError


@pytest.fixture(autouse=True)
def _fresh_manager(monkeypatch):
    """Keep the process-wide manager out of this module's way."""
    monkeypatch.setattr(dynamic_memory_store, "_dynamic_manager", None)


def test_create_memory_store_opens_no_engine_of_its_own(monkeypatch):
    create_engine_mock = Mock(wraps=sqlalchemy.create_engine)
    monkeypatch.setattr(sqlalchemy, "create_engine", create_engine_mock)

    # With no authority readable in this context, the call fails closed rather
    # than falling back to whatever model the hub happens to hold.
    with pytest.raises(MemoryUnavailableError):
        memory_utils.create_memory_store()

    assert create_engine_mock.call_count == 0


def test_create_memory_store_returns_the_admitted_store(monkeypatch):
    published = object()
    monkeypatch.setattr(
        dynamic_memory_store.DynamicMemoryStoreManager,
        "get_memory_store",
        lambda _self: published,
    )

    assert memory_utils.create_memory_store() is published
