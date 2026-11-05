"""A KB request's usage gate is told the models the request then calls.

The entry point resolves the models under a ``ModelResolutionPin`` to learn
the ids it gates on, and runs the pipeline under the same pin. Between the
two, the state resolution reads -- the collection's binding, the model hub --
is changed here, the way a concurrent request or an operator could change it:
the pipeline must still call what the gate was told about.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig, RerankModelConfig
from xagent.core.model.rerank.dashscope import DashscopeRerank
from xagent.core.model.storage.error import ModelNotFoundError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DenseSearchResponse,
    IndexStatus,
    SearchConfig,
    SearchResult,
    SearchType,
)
from xagent.core.tools.core.RAG_tools.management import collection_manager
from xagent.core.tools.core.RAG_tools.pipelines import (
    document_ingestion,
    document_search,
)
from xagent.core.tools.core.RAG_tools.utils import model_resolver
from xagent.core.tools.core.RAG_tools.utils.model_resolution_pin import (
    ModelResolutionPin,
)

PLATFORM = "platform/text-embedding-v4"


def _embedding(model_id: str) -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        id=model_id,
        model_name="text-embedding-v4",
        model_provider="openai",
        api_key="test-key",
    )


class _World:
    """What resolution reads, and what the pipeline ends up calling."""

    def __init__(self) -> None:
        self.bound: str | None = None  # the collection's bound model
        self.rows: dict[str, Any] = {}  # hub lookup id -> config
        self.embedded_with: list[str] = []
        self.reranked_with: list[str] = []


class _FakeEmbedding(BaseEmbedding):
    def __init__(self, world: _World, model_id: str) -> None:
        self._world = world
        self._model_id = model_id

    def encode(self, text: Any, dimension: Any = None, instruct: Any = None) -> Any:
        self._world.embedded_with.append(self._model_id)
        return [1.0] if isinstance(text, str) else [[1.0] for _ in text]

    def get_dimension(self) -> int:
        return 1

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


class _FakeRerank(DashscopeRerank):
    def __init__(self, world: _World, model_id: str) -> None:
        self._world = world
        self.model_id = model_id

    def compress_with_scores(self, documents: Any, query: str) -> Any:
        self._world.reranked_with.append(self.model_id)
        return [(document, 1.0) for document in documents]


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> _World:
    world = _World()

    class _Hub:
        def load(self, model_id: str, name_match_scope: Any = None) -> Any:
            if model_id not in world.rows:
                raise ModelNotFoundError(model_id)
            return world.rows[model_id]

    @contextmanager
    def managed_hub(_model_type_name: str) -> Any:
        yield _Hub(), None

    class _Maintenance:
        def resolve_effective_embedding_model_sync(
            self, collection: str, requested: str | None
        ) -> str:
            if world.bound:
                return world.bound
            if requested:
                return requested
            raise ValueError(f"No embedding model for collection '{collection}'")

    monkeypatch.setattr(model_resolver, "_managed_model_hub", managed_hub)
    monkeypatch.setattr(
        model_resolver,
        "create_embedding_adapter",
        lambda cfg: _FakeEmbedding(world, cfg.usage_model_id),
    )
    monkeypatch.setattr(
        model_resolver,
        "create_rerank_adapter",
        lambda cfg: _FakeRerank(world, cfg.usage_model_id),
    )
    monkeypatch.setattr(
        collection_manager,
        "_get_maintenance_compatibility_facade",
        lambda: _Maintenance(),
    )
    for name in (
        "DASHSCOPE_EMBEDDING_MODEL",
        "DASHSCOPE_EMBEDDING_API_KEY",
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_RERANK_MODEL",
        "DASHSCOPE_RERANK_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    return world


@pytest.fixture
def dense_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = SearchResult(
        doc_id="d", chunk_id="c", text="t", score=1.0, parse_hash="p", model_tag="m"
    )
    monkeypatch.setattr(
        document_search,
        "search_dense",
        lambda **_kwargs: DenseSearchResponse(
            results=[hit], total_count=1, index_status=IndexStatus.INDEX_READY
        ),
    )


def _search(pin: ModelResolutionPin, config: SearchConfig) -> tuple[str, ...]:
    """The ids the search entry point gates on, resolved under ``pin``."""
    return pin.run(
        document_search.resolve_search_model_ids,
        "kb",
        config=config,
        user_id=1,
        is_admin=False,
    )


def _run_search(pin: ModelResolutionPin, config: SearchConfig) -> Any:
    return pin.run(
        document_search._search_documents_impl,
        "kb",
        "query",
        config=config,
        progress_manager=MagicMock(),
        user_id=1,
        is_admin=False,
    )


def test_search_calls_the_bound_model_the_gate_was_told_about(world, dense_hits):
    # The request names the platform model, the collection is bound to the
    # team's own: the bound one is called, so the bound one is gated.
    world.bound = "team-embedding"
    world.rows = {
        "team-embedding": _embedding("team-embedding"),
        PLATFORM: _embedding(PLATFORM),
        "platform/rerank": RerankModelConfig(
            id="platform/rerank", model_name="qwen3-rerank", api_key="k"
        ),
    }
    config = SearchConfig(
        search_type=SearchType.DENSE,
        embedding_model_id=PLATFORM,
        rerank_model_id="platform/rerank",
    )
    pin = ModelResolutionPin()

    model_ids = _search(pin, config)
    # Gating resolves; it calls nothing.
    assert world.embedded_with == [] and world.reranked_with == []
    # Rebound and repointed between the gate and the search.
    world.bound = PLATFORM
    world.rows["team-embedding"] = _embedding(PLATFORM)
    world.rows["platform/rerank"] = RerankModelConfig(
        id="team-rerank", model_name="qwen3-rerank", api_key="k"
    )
    result = _run_search(pin, config)

    assert result.status == "success", result.message
    assert model_ids == ("team-embedding", "platform/rerank")
    assert world.embedded_with == ["team-embedding"]
    assert world.reranked_with == ["platform/rerank"]


def test_search_gates_the_model_a_bare_name_resolves_to(world, dense_hits):
    # The hub resolves the bare name to the platform row (by visible name).
    world.rows = {"text-embedding-v4": _embedding(PLATFORM)}
    config = SearchConfig(
        search_type=SearchType.DENSE, embedding_model_id="text-embedding-v4"
    )
    pin = ModelResolutionPin()

    model_ids = _search(pin, config)
    world.rows["text-embedding-v4"] = _embedding("someone-elses-embedding")
    _run_search(pin, config)

    assert model_ids == (PLATFORM,)
    assert world.embedded_with == [PLATFORM]


def test_an_environment_model_is_gated_with_no_id(world, dense_hits, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.setenv("DASHSCOPE_EMBEDDING_API_KEY", "env-key")
    config = SearchConfig(
        search_type=SearchType.DENSE, embedding_model_id="text-embedding-v4"
    )
    pin = ModelResolutionPin()

    model_ids = _search(pin, config)
    _run_search(pin, config)

    assert model_ids == ()
    # Recorded unpriced: the call carries no model id either.
    assert world.embedded_with == [""]


def test_a_sparse_search_gates_no_embedding_model(world):
    world.rows = {PLATFORM: _embedding(PLATFORM)}
    config = SearchConfig(search_type=SearchType.SPARSE, embedding_model_id=PLATFORM)

    assert _search(ModelResolutionPin(), config) == ()


def test_a_resolution_that_failed_for_the_gate_fails_the_search(world, dense_hits):
    config = SearchConfig(search_type=SearchType.DENSE, embedding_model_id=PLATFORM)
    pin = ModelResolutionPin()

    model_ids = _search(pin, config)
    # The row appears after the gate was told there was nothing to call.
    world.rows[PLATFORM] = _embedding(PLATFORM)
    result = _run_search(pin, config)

    assert model_ids == ()
    assert result.status == "error"
    assert world.embedded_with == []


@pytest.fixture
def ingestion_stops_after_resolving(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Run ingestion only up to the model it resolved, and report that model."""
    initialized: list[str] = []

    def initialize(collection_name: str, embedding_model_id: str) -> None:
        initialized.append(embedding_model_id)
        raise RuntimeError("stop after resolving")

    monkeypatch.setattr(
        document_ingestion, "validate_document_processing_sync", lambda **_: None
    )
    monkeypatch.setattr(
        document_ingestion, "initialize_collection_embedding_sync", initialize
    )
    monkeypatch.setattr(
        document_ingestion, "_record_ingestion_status", lambda *_a, **_k: None
    )
    return initialized


@pytest.mark.parametrize(
    "bound,requested,rows,expected",
    [
        # Bound to the team's model; the request names the platform one.
        (
            "team-embedding",
            PLATFORM,
            {"team-embedding": "team-embedding", PLATFORM: PLATFORM},
            "team-embedding",
        ),
        # Not bound yet; a bare name the hub resolves to the platform row.
        (None, "text-embedding-v4", {"text-embedding-v4": PLATFORM}, PLATFORM),
    ],
    ids=["bound-elsewhere", "bare-name"],
)
def test_ingestion_embeds_with_the_model_the_gate_was_told_about(
    world, ingestion_stops_after_resolving, bound, requested, rows, expected
):
    world.bound = bound
    world.rows = {key: _embedding(model_id) for key, model_id in rows.items()}
    config = {"embedding_model_id": requested}
    pin = ModelResolutionPin()

    model_ids = pin.run(
        document_ingestion.resolve_ingestion_model_ids,
        "kb",
        ingestion_config=config,
        user_id=1,
        is_admin=False,
    )
    world.bound = "changed-binding"
    world.rows = {key: _embedding("changed-row") for key in [*rows, "changed-binding"]}
    pin.run(
        document_ingestion._process_document_impl,
        "kb",
        "/tmp/notes.txt",
        config=document_ingestion.coerce_ingestion_config(config),
        progress_manager=MagicMock(),
        user_id=1,
        is_admin=False,
    )

    assert model_ids == (expected,)
    assert ingestion_stops_after_resolving == [expected]


def test_threads_racing_on_one_resolution_all_use_the_first_that_lands():
    import threading

    from xagent.core.tools.core.RAG_tools.utils.model_resolution_pin import (
        pinned_resolution,
    )

    pin = ModelResolutionPin()
    both_computing = threading.Barrier(2)
    seen: list[str] = []

    def resolve(answer: str) -> None:
        def compute() -> str:
            both_computing.wait(timeout=5)  # neither has landed yet
            if answer == "late":
                first_landed.wait(timeout=5)
            return answer

        seen.append(pin.run(pinned_resolution, "key", compute))
        if answer == "early":
            first_landed.set()

    first_landed = threading.Event()
    threads = [threading.Thread(target=resolve, args=(a,)) for a in ("early", "late")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert seen == ["early", "early"]
