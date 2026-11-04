from __future__ import annotations

import pytest
from chromadb import Client
from chromadb.config import Settings

from xagent.providers.vector_store.chroma import ChromaVectorStore


@pytest.fixture(scope="session")
def chroma_client(tmp_path_factory):
    persist_dir = str(tmp_path_factory.getbasetemp() / "chroma_persist")
    settings = Settings(persist_directory=persist_dir, anonymized_telemetry=False)
    client = Client(settings=settings)
    yield client


@pytest.fixture(scope="function")
def store(chroma_client):
    vs = ChromaVectorStore(client=chroma_client, collection_name="test_memory")
    yield vs
    vs.clear()


def test_add_and_search_vectors(store: ChromaVectorStore):
    # Prepare sample vectors and associated metadata
    vectors = [[0.1 * i for i in range(10)], [0.2 * i for i in range(10)]]
    metadatas = [{"note_id": "note1"}, {"note_id": "note2"}]

    # Add vectors to the store
    ids = store.add_vectors(vectors, metadatas=metadatas)
    assert len(ids) == 2

    # Search using a query vector
    query_vector = [0.15 * i for i in range(10)]
    results = store.search_vectors(query_vector, top_k=2)
    assert len(results) == 2

    # Ensure result structure is correct
    assert all("id" in hit and "score" in hit and "metadata" in hit for hit in results)


def test_delete_vectors(store: ChromaVectorStore):
    # Add a vector and get its ID
    vector = [[0.5 for _ in range(10)]]
    ids = store.add_vectors(vector)
    assert len(ids) == 1

    # Delete the vector
    success = store.delete_vectors(ids)
    assert success

    # Search should return no results after deletion
    results = store.search_vectors(query_vector=[0.5] * 10, top_k=1)
    assert results == []


def test_clear_store(store: ChromaVectorStore):
    # Add a vector and then clear the store
    store.add_vectors([[0.3 for _ in range(10)]])
    store.clear()

    # After clearing, search should return no results
    results = store.search_vectors([0.3 for _ in range(10)], top_k=1)
    assert results == []
