"""Tests for generation-scoped ingestion locks (PR #202)."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from xagent.core.tools.core.RAG_tools.utils import generation_lock
from xagent.core.tools.core.RAG_tools.utils.generation_lock import (
    generation_ingestion_lock,
    generation_lock_key,
)


@pytest.fixture(autouse=True)
def _clear_thread_lock_registry() -> None:
    generation_lock._thread_locks.clear()
    yield
    generation_lock._thread_locks.clear()


def test_generation_lock_key_includes_scope_not_source_path() -> None:
    """Lock key is tied to doc generation scope, not file path."""
    key_a = generation_lock_key("col", "doc-1", "parse-1", 42, False)
    key_b = generation_lock_key("col", "doc-1", "parse-1", 42, False)
    key_c = generation_lock_key("col", "doc-1", "parse-2", 42, False)
    key_d = generation_lock_key("col", "doc-1", "parse-1", 99, False)
    assert key_a == key_b
    assert key_a != key_c
    assert key_a != key_d


def test_generation_ingestion_lock_serialises_same_scope() -> None:
    """Two threads for the same generation scope cannot overlap."""
    execution_log: list[str] = []

    def _worker() -> None:
        with generation_ingestion_lock(
            "col", "doc1", "hash1", user_id=1, is_admin=False
        ):
            execution_log.append("enter")
            time.sleep(0.1)
            execution_log.append("exit")

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert execution_log == ["enter", "exit", "enter", "exit"]


def test_generation_ingestion_lock_allows_different_scopes() -> None:
    """Different (doc_id, parse_hash) pairs do not block each other."""
    entered = threading.Event()
    gate = threading.Event()

    def _worker(doc_id: str) -> None:
        with generation_ingestion_lock(
            "col", doc_id, "hash1", user_id=1, is_admin=False
        ):
            entered.set()
            gate.wait(timeout=5)

    t1 = threading.Thread(target=_worker, args=("doc-a",))
    t2 = threading.Thread(target=_worker, args=("doc-b",))
    t1.start()
    assert entered.wait(timeout=5)
    entered.clear()
    t2.start()
    both = entered.wait(timeout=2)
    gate.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert both


def test_generation_ingestion_lock_allows_different_user_ids() -> None:
    """Same doc scope for different users must not block each other (tenancy)."""
    entered = threading.Event()
    gate = threading.Event()

    def _worker(user_id: int) -> None:
        with generation_ingestion_lock(
            "col", "doc-1", "hash-1", user_id=user_id, is_admin=False
        ):
            entered.set()
            gate.wait(timeout=5)

    t1 = threading.Thread(target=_worker, args=(1,))
    t2 = threading.Thread(target=_worker, args=(2,))
    t1.start()
    assert entered.wait(timeout=5)
    entered.clear()
    t2.start()
    both = entered.wait(timeout=2)
    gate.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert both


@patch("xagent.core.tools.core.RAG_tools.utils.generation_lock.FileLock")
def test_generation_ingestion_lock_reentrant_nested(mock_file_lock: object) -> None:
    """Nested acquire for the same scope only takes the file lock once."""
    mock_file_lock.return_value.acquire.return_value = None
    mock_file_lock.return_value.release.return_value = None

    with generation_ingestion_lock("col", "doc1", "hash1", user_id=None, is_admin=True):
        with generation_ingestion_lock(
            "col", "doc1", "hash1", user_id=None, is_admin=True
        ):
            pass

    assert mock_file_lock.return_value.acquire.call_count == 1
    assert mock_file_lock.return_value.release.call_count == 1
