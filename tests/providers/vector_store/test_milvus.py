from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

pytest.importorskip("pymilvus")

from xagent.providers.vector_store import (  # noqa: E402
    MilvusConnectionManager,
    MilvusVectorStore,
)
from xagent.providers.vector_store.milvus import (  # noqa: E402
    get_client,
    get_client_from_env,
)


class FakeMilvusClient:
    def __init__(self, *_: Any, **__: Any):
        self._collections: Dict[str, Dict[str, Any]] = {}

    def has_collection(self, collection_name: str, **__: Any) -> bool:
        return collection_name in self._collections

    def create_collection(
        self,
        collection_name: str,
        dimension: int | None = None,
        **__: Any,
    ) -> None:
        self._collections[collection_name] = {"dimension": dimension, "rows": []}

    def insert(
        self, collection_name: str, data: List[Dict[str, Any]], **__: Any
    ) -> Dict:
        table = self._collections[collection_name]
        table["rows"].extend(data)
        return {"insert_count": len(data)}

    def search(
        self,
        collection_name: str,
        data: List[List[float]] | None = None,
        limit: int = 10,
        **__: Any,
    ) -> List[List[Dict[str, Any]]]:
        table = self._collections[collection_name]
        query = data[0] if data else []

        def _dist(a: List[float], b: List[float]) -> float:
            return float(sum((x - y) ** 2 for x, y in zip(a, b)))

        rows = sorted(table["rows"], key=lambda r: _dist(r["vector"], query))
        return [
            [
                {
                    "id": row["id"],
                    "distance": _dist(row["vector"], query),
                    "entity": {"metadata": row.get("metadata", {})},
                }
                for row in rows[:limit]
            ]
        ]

    def delete(
        self,
        collection_name: str,
        ids: List[str] | None = None,
        **__: Any,
    ) -> Dict[str, int]:
        table = self._collections[collection_name]
        before = len(table["rows"])
        if ids:
            table["rows"] = [row for row in table["rows"] if row["id"] not in ids]
        return {"delete_count": before - len(table["rows"])}

    def truncate_collection(self, collection_name: str, **__: Any) -> None:
        self._collections[collection_name]["rows"] = []


@pytest.fixture
def fake_milvus_client():
    return FakeMilvusClient()


@pytest.fixture
def patch_milvus_client(fake_milvus_client):
    with patch(
        "xagent.providers.vector_store.milvus._import_milvus_client_class"
    ) as mock_import:
        mock_import.return_value = lambda **kwargs: fake_milvus_client
        yield fake_milvus_client


def test_add_and_search_vectors(patch_milvus_client):
    store = MilvusVectorStore(uri="http://localhost:19530", collection_name="test_vs")
    ids = store.add_vectors(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"kind": "a"}, {"kind": "b"}],
    )
    assert len(ids) == 2

    results = store.search_vectors(query_vector=[1.0, 0.1], top_k=2)
    assert len(results) == 2
    assert all("id" in r and "score" in r and "metadata" in r for r in results)


def test_search_with_filters(patch_milvus_client):
    store = MilvusVectorStore(uri="http://localhost:19530", collection_name="test_vs")
    store.add_vectors(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        ids=["a", "b"],
        metadatas=[{"tenant": "x"}, {"tenant": "y"}],
    )

    results = store.search_vectors(
        query_vector=[1.0, 0.0],
        top_k=5,
        filters={"tenant": "x"},
    )
    assert len(results) == 1
    assert results[0]["id"] == "a"


def test_delete_and_clear(patch_milvus_client):
    store = MilvusVectorStore(uri="http://localhost:19530", collection_name="test_vs")
    ids = store.add_vectors(vectors=[[1.0, 0.0], [0.0, 1.0]])
    assert store.delete_vectors([ids[0]]) is True
    assert len(store.search_vectors([1.0, 0.0], top_k=10)) == 1

    store.clear()
    assert store.search_vectors([1.0, 0.0], top_k=10) == []


def test_connection_manager_from_env(patch_milvus_client):
    manager = MilvusConnectionManager()

    with patch.dict(
        os.environ,
        {
            "TEST_MILVUS_URI": "http://localhost:19530",
            "TEST_MILVUS_TOKEN": "token",
            "TEST_MILVUS_DB": "default",
        },
    ):
        client = manager.get_client_from_env(
            uri_env_var="TEST_MILVUS_URI",
            token_env_var="TEST_MILVUS_TOKEN",
            db_name_env_var="TEST_MILVUS_DB",
        )
        assert client is not None


def test_connection_manager_from_env_missing_uri(patch_milvus_client):
    manager = MilvusConnectionManager()
    with pytest.raises(
        KeyError, match="Environment variable TEST_MISSING_URI is not set"
    ):
        manager.get_client_from_env(uri_env_var="TEST_MISSING_URI")


def test_convenience_functions(patch_milvus_client):
    client = get_client(uri="http://localhost:19530")
    assert client is not None

    with patch.dict(os.environ, {"TEST_MILVUS_URI": "http://localhost:19530"}):
        env_client = get_client_from_env(uri_env_var="TEST_MILVUS_URI")
        assert env_client is not None


class Client:
    def __init__(self, **settings: Any):
        self.settings = settings


@pytest.fixture
def shared_clients(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    """Clear the process-wide clients; the list records each connect's settings."""
    monkeypatch.setattr(MilvusConnectionManager, "_shared_clients", {})
    monkeypatch.setenv("MILVUS_URI", "http://milvus.test:19530")
    monkeypatch.setenv("MILVUS_TOKEN", "token")
    monkeypatch.setenv("MILVUS_DB_NAME", "db")
    connects: List[Dict[str, Any]] = []

    def connect(**settings: Any) -> Client:
        connects.append(settings)
        return Client(**settings)

    with patch(
        "xagent.providers.vector_store.milvus._import_milvus_client_class",
        return_value=connect,
    ):
        yield connects


def test_managers_share_a_client_per_uri_token_and_database(shared_clients):
    first = MilvusConnectionManager().get_shared_client_from_env()

    assert MilvusConnectionManager().get_shared_client_from_env() is first
    assert first.settings == {
        "uri": "http://milvus.test:19530",
        "token": "token",
        "db_name": "db",
    }
    assert len(shared_clients) == 1
    for name, value in (
        ("MILVUS_DB_NAME", "other"),
        ("MILVUS_TOKEN", "other"),
        ("MILVUS_URI", "http://other:19530"),
    ):
        with patch.dict(os.environ, {name: value}):
            elsewhere = MilvusConnectionManager().get_shared_client_from_env()
        assert elsewhere is not first
    assert len(shared_clients) == 4


def test_a_different_manager_does_not_get_the_first_client(shared_clients):
    class Other(MilvusConnectionManager):
        pass

    first = MilvusConnectionManager().get_shared_client_from_env()
    other = Other().get_shared_client_from_env()

    assert other is not first
    assert Other().get_shared_client_from_env() is other
    assert len(shared_clients) == 2


def test_a_failed_connect_is_not_cached(shared_clients):
    manager = MilvusConnectionManager()
    attempts = []

    def connect(**settings: Any) -> Client:
        attempts.append(settings)
        if len(attempts) == 1:
            raise ConnectionError("unreachable")
        return Client(**settings)

    with patch(
        "xagent.providers.vector_store.milvus._import_milvus_client_class",
        return_value=connect,
    ):
        with pytest.raises(ConnectionError):
            manager.get_shared_client_from_env()
        assert (
            manager.get_shared_client_from_env() is manager.get_shared_client_from_env()
        )
    assert len(attempts) == 2


def test_the_shared_client_needs_a_uri(shared_clients, monkeypatch):
    monkeypatch.delenv("MILVUS_URI")
    with pytest.raises(KeyError, match="MILVUS_URI is not set"):
        MilvusConnectionManager().get_shared_client_from_env()
    monkeypatch.setenv("MILVUS_URI", " ")
    with pytest.raises(ValueError, match="MILVUS_URI is empty"):
        MilvusConnectionManager().get_shared_client_from_env()
    assert shared_clients == []


def test_a_cached_client_is_returned_while_another_thread_connects(shared_clients):
    cached = MilvusConnectionManager().get_shared_client_from_env()
    returned: List[Any] = []

    with MilvusConnectionManager._shared_clients_lock:
        worker = threading.Thread(
            target=lambda: returned.append(
                MilvusConnectionManager().get_shared_client_from_env()
            )
        )
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
    assert returned == [cached]


def test_concurrent_first_use_connects_once(shared_clients):
    def connect(**settings: Any) -> Client:
        shared_clients.append(settings)
        time.sleep(0.05)
        return Client(**settings)

    with patch(
        "xagent.providers.vector_store.milvus._import_milvus_client_class",
        return_value=connect,
    ):
        with ThreadPoolExecutor(6) as pool:
            clients = list(
                pool.map(
                    lambda _: MilvusConnectionManager().get_shared_client_from_env(),
                    range(6),
                )
            )

    assert len({id(client) for client in clients}) == 1
    assert len(shared_clients) == 1
