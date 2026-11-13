"""Tests for the embedding migration's copy command."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

import pyarrow as pa
import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.migrations.embedding import embed as mod
from xagent.migrations.embedding import scope
from xagent.migrations.embedding.scope import MigrationCliError


class _StubAdapter(BaseEmbedding):
    def __init__(self) -> None:
        self.calls: List[Union[str, List[str]]] = []

    def encode(  # type: ignore[override]
        self,
        text: Union[str, List[str]],
        dimension: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        self.calls.append(text)
        if isinstance(text, str):
            return [0.5, 0.5]
        return [[0.5, 0.5] for _ in text]

    def get_dimension(self) -> int:
        return 2

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


class _Table:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows
        names = list(rows[0]) if rows else ["doc_id", "chunk_id", "text", "vector"]
        self.schema = pa.schema([(name, pa.string()) for name in names])

    def to_batches(self, *, filter: str, batch_size: int) -> Iterator[Any]:
        if self.rows:
            yield pa.RecordBatch.from_pylist(self.rows)

    def search(self) -> "_Query":
        return _Query(self)

    def close(self) -> None:  # pragma: no cover
        pass


class _Query:
    """The slice of LanceDB's query builder ``iter_rows`` uses."""

    def __init__(self, table: Any) -> None:
        self.table = table
        self.filter = ""
        self.columns: List[str] = []

    def where(self, filter: str) -> "_Query":
        self.filter = filter
        return self

    def select(self, columns: List[str]) -> "_Query":
        self.columns = list(columns)
        return self

    def limit(self, value: Any) -> "_Query":
        return self

    def to_batches(self, batch_size: int) -> Iterator[Any]:
        for batch in self.table.to_batches(filter=self.filter, batch_size=batch_size):
            yield pa.RecordBatch.from_pylist(
                [{k: row.get(k) for k in self.columns} for row in batch.to_pylist()]
            )


class _Conn:
    def __init__(self, tables: Dict[str, List[Dict[str, Any]]]) -> None:
        self.tables = {name: _Table(rows) for name, rows in tables.items()}

    def open_table(self, name: str) -> _Table:
        if name not in self.tables:
            raise ValueError(f"Table '{name}' not found")
        return self.tables[name]

    def create_table(self, name: str, **kwargs: Any) -> None:
        raise AssertionError(f"must not create {name}")

    def list_tables(self) -> List[str]:
        return list(self.tables)


def _row(**overrides: Any) -> Dict[str, Any]:
    row = {
        "doc_id": "doc-1",
        "chunk_id": "c1",
        "parse_hash": "hash-1",
        "text": "hello",
        "chunk_hash": "ch-1",
        "metadata": None,
        "user_id": 7,
        "vector": [0.5, 0.5],
    }
    row.update(overrides)
    return row


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    conn: _Conn,
    *,
    engine: Optional[str] = "lancedb",
    bound: Optional[str] = "old",
    adapter: Optional[_StubAdapter] = None,
) -> List[Dict[str, Any]]:
    writes: List[Dict[str, Any]] = []
    monkeypatch.setattr(mod, "connection", lambda: conn)
    monkeypatch.setattr(
        mod,
        "collection_info",
        lambda name: __import__("types").SimpleNamespace(embedding_model_id=bound),
    )
    monkeypatch.setattr(mod, "collection_engine", lambda info: engine)
    monkeypatch.setattr(
        mod,
        "resolve_target_model",
        lambda model_id: (
            EmbeddingModelConfig(id=model_id, model_name="stub", dimension=2),
            adapter or _StubAdapter(),
        ),
    )
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["kb"])
    monkeypatch.setattr(mod, "embeddings_table", lambda model_id, conn: "target")

    def _doc_ids(name: str, model_id: str, *, conn: Any = None) -> Any:
        rows = conn.tables["embeddings_old"].rows
        ids = sorted({row["doc_id"] for row in rows if row.get("doc_id")})
        unaddressable = sum(
            1 for row in rows if not row.get("doc_id") or not row.get("chunk_id")
        )
        return ids, unaddressable

    def _chunks(
        name: str, model_id: str, *, conn: Any = None, doc_id: Any = None
    ) -> Any:
        for row in conn.tables["embeddings_old"].rows:
            if doc_id is not None and row.get("doc_id") != doc_id:
                continue
            yield __import__("types").SimpleNamespace(
                doc_id=str(row.get("doc_id") or ""),
                chunk_id=str(row.get("chunk_id") or ""),
                parse_hash=row.get("parse_hash") or "",
                text=str(row.get("text") or ""),
                chunk_hash=row.get("chunk_hash") or "",
                metadata=row.get("metadata"),
                user_id=row.get("user_id"),
            )

    monkeypatch.setattr(mod, "iter_source_doc_ids", _doc_ids)
    monkeypatch.setattr(mod, "iter_source_chunks", _chunks)

    def _write(**kwargs: Any) -> None:
        writes.append(kwargs)
        target = conn.tables.setdefault("embeddings_new", _Table([]))
        target.rows = target.rows + [
            {"chunk_id": item.chunk_id, "vector": item.vector}
            for item in kwargs["embeddings"]
        ]

    monkeypatch.setattr(mod, "write_vectors_to_db", _write)
    monkeypatch.setattr(mod, "_ensure_indexes", lambda *args, **kwargs: None)
    return writes


def test_a_missing_chunk_is_copied_under_its_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes = _wire(monkeypatch, _Conn({"embeddings_old": [_row()]}))

    report = mod.embed_collections("new")

    assert len(writes) == 1
    written = writes[0]["embeddings"][0]
    assert written.doc_id == "doc-1"
    assert written.chunk_id == "c1"
    assert written.parse_hash == "hash-1"
    assert written.text == "hello"
    assert written.chunk_hash == "ch-1"
    assert written.model == "new"
    assert writes[0]["user_id"] == 7
    assert writes[0]["create_index"] is False
    item = report.collections[0]
    assert item.documents_embedded == 1
    assert item.chunks_embedded == 1
    assert report.failed is False


def test_a_chunk_the_target_already_holds_is_not_re_embedded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    conn = _Conn(
        {
            "embeddings_old": [_row()],
            "embeddings_new": [{"chunk_id": "c1", "vector": [0.5, 0.5]}],
        }
    )
    writes = _wire(monkeypatch, conn, adapter=adapter)

    report = mod.embed_collections("new")

    assert writes == []
    assert adapter.calls == []
    assert report.collections[0].documents_unchanged == 1
    assert report.collections[0].chunks_pending == 0


@pytest.mark.parametrize(
    "vector",
    [None, [0.5], [0.0, 0.0], [float("inf"), 1.0], [float("nan"), 1.0]],
)
def test_a_target_row_without_a_usable_vector_is_still_pending(
    monkeypatch: pytest.MonkeyPatch, vector: Any
) -> None:
    adapter = _StubAdapter()
    conn = _Conn(
        {
            "embeddings_old": [_row()],
            "embeddings_new": [{"chunk_id": "c1", "vector": vector}],
        }
    )
    writes = _wire(monkeypatch, conn, adapter=adapter)

    report = mod.embed_collections("new")

    assert len(writes) == 1
    assert adapter.calls
    assert report.collections[0].chunks_pending == 1


def test_a_dry_run_neither_calls_the_model_nor_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    writes = _wire(monkeypatch, _Conn({"embeddings_old": [_row()]}), adapter=adapter)

    report = mod.embed_collections("new", dry_run=True)

    assert writes == []
    assert adapter.calls == []
    assert report.collections[0].documents_embedded == 1
    assert report.collections[0].chunks_pending == 1
    assert report.collections[0].chunks_embedded == 0


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"user_id": None}, scope.SKIP_NO_OWNER),
        ({"text": ""}, scope.SKIP_NO_TEXT),
        ({"doc_id": ""}, scope.SKIP_NO_DOC_ID),
    ],
)
def test_unusable_rows_are_counted_and_not_written(
    monkeypatch: pytest.MonkeyPatch, overrides: Dict[str, Any], reason: str
) -> None:
    writes = _wire(monkeypatch, _Conn({"embeddings_old": [_row(**overrides)]}))

    report = mod.embed_collections("new")

    assert writes == []
    assert report.collections[0].skipped == {reason: 1}


def test_a_partial_write_is_a_failure_not_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn({"embeddings_old": [_row(chunk_id="c1"), _row(chunk_id="c2")]})
    _wire(monkeypatch, conn)

    def _drop_one(**kwargs: Any) -> None:
        conn.tables["embeddings_new"] = _Table(
            [{"chunk_id": "c1", "vector": [0.5, 0.5]}]
        )

    monkeypatch.setattr(mod, "write_vectors_to_db", _drop_one)

    report = mod.embed_collections("new")

    item = report.collections[0]
    assert report.failed is True
    assert item.failures[0].doc_id == "doc-1"
    assert "missing from the target model" in item.failures[0].error


def test_a_document_failure_does_not_stop_the_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            "embeddings_old": [
                _row(doc_id="a", chunk_id="a1"),
                _row(doc_id="b", chunk_id="b1"),
            ]
        }
    )
    _wire(monkeypatch, conn)
    calls = {"n": 0}

    def _fail_first(**kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("provider exploded")
        conn.tables["embeddings_new"] = _Table(
            [{"chunk_id": "b1", "vector": [0.5, 0.5]}]
        )

    monkeypatch.setattr(mod, "write_vectors_to_db", _fail_first)

    report = mod.embed_collections("new")

    item = report.collections[0]
    assert [failure.doc_id for failure in item.failures] == ["a"]
    assert item.documents_embedded == 1


def test_consecutive_failures_abort_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = [_row(doc_id=f"d{index}", chunk_id=f"c{index}") for index in range(6)]
    _wire(monkeypatch, _Conn({"embeddings_old": docs}))

    def _always_fail(**kwargs: Any) -> None:
        raise RuntimeError("no credentials")

    monkeypatch.setattr(mod, "write_vectors_to_db", _always_fail)

    report = mod.embed_collections("new")

    assert report.aborted is True
    assert len(report.collections[0].failures) == mod.MAX_CONSECUTIVE_DOCUMENT_FAILURES


def test_one_collection_crashing_does_not_stop_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn({"embeddings_old": [_row()]})
    _wire(monkeypatch, conn)
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["a", "b"])

    def _info(name: str) -> Any:
        if name == "a":
            raise OSError("metadata gone")
        return __import__("types").SimpleNamespace(embedding_model_id="old")

    monkeypatch.setattr(mod, "collection_info", _info)

    report = mod.embed_collections("new")

    assert [item.name for item in report.collections] == ["a", "b"]
    assert report.collections[0].failures[0].doc_id == "*"
    assert report.collections[1].documents_embedded == 1


def test_another_engine_is_skipped_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes = _wire(monkeypatch, _Conn({"embeddings_old": [_row()]}), engine="milvus")

    report = mod.embed_collections("new")

    assert report.collections[0].collection_skip == mod.SKIP_OTHER_ENGINE
    assert writes == []


def test_an_unbound_collection_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _wire(monkeypatch, _Conn({}), bound=None)

    report = mod.embed_collections("new")

    assert report.collections[0].collection_skip == "no-source-model"
    assert writes == []


def test_an_unknown_target_model_never_falls_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, _Conn({"embeddings_old": [_row()]}))

    def _raise(model_id: str) -> Any:
        raise MigrationCliError(f"'{model_id}' is not configured")

    monkeypatch.setattr(mod, "resolve_target_model", _raise)

    with pytest.raises(MigrationCliError, match="not configured"):
        mod.embed_collections("missing-model")


def test_the_resolver_rejects_a_fallback_with_another_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import embedding_switch
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    monkeypatch.setattr(
        embedding_switch,
        "resolve_configured_model",
        lambda model_id: (model_id, 2),
    )
    monkeypatch.setattr(
        model_resolver,
        "resolve_embedding_adapter",
        lambda model_id: (
            EmbeddingModelConfig(id="env-default", model_name="x", dimension=2),
            _StubAdapter(),
        ),
    )

    with pytest.raises(MigrationCliError, match="pass the configured id"):
        mod.resolve_target_model("new")


def test_the_rate_limiter_spaces_calls_at_the_configured_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 100.0}
    sleeps: List[float] = []
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: sleeps.append(seconds))
    limiter = mod.RateLimiter(2.0)

    limiter.acquire(2)
    limiter.acquire(2)

    assert sleeps == [1.0]


def test_the_rate_limiter_never_sleeps_without_a_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mod.time, "sleep", lambda seconds: pytest.fail(f"slept {seconds}")
    )

    mod.RateLimiter(0.0).acquire(10)


def test_paced_embedding_charges_one_unit_per_text() -> None:
    limiter = mod.RateLimiter(0.0)
    seen: List[int] = []
    limiter.acquire = seen.append  # type: ignore[method-assign]
    paced = mod.PacedEmbedding(_StubAdapter(), limiter)

    paced.encode("one")
    paced.encode(["a", "b", "c"])

    assert seen == [1, 3]


def test_one_limiter_paces_every_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, _Conn({"embeddings_old": [_row()]}))
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["a", "b"])
    charges: List[int] = []
    real = mod.RateLimiter(1.0)

    def _acquire(count: int) -> None:
        charges.append(count)

    real.acquire = _acquire  # type: ignore[method-assign]
    monkeypatch.setattr(mod, "RateLimiter", lambda rate: real)
    monkeypatch.setattr(mod, "target_chunk_ids", lambda *args, **kwargs: set())

    mod.embed_collections("new")

    assert len(charges) == 2


def test_the_configured_batch_size_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row(chunk_id=f"c{index}") for index in range(5)]
    writes = _wire(monkeypatch, _Conn({"embeddings_old": rows}))

    mod.embed_collections("new", batch_size=2)

    assert [len(call["embeddings"]) for call in writes] == [2, 2, 1]


def test_the_async_path_encodes_one_text_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _StubAdapter()
    _wire(
        monkeypatch,
        _Conn({"embeddings_old": [_row(chunk_id="c1"), _row(chunk_id="c2")]}),
        adapter=adapter,
    )

    mod.embed_collections("new", use_async=True)

    assert len(adapter.calls) == 2
    assert all(isinstance(call, str) for call in adapter.calls)


def test_a_dropped_async_chunk_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Dropping(_StubAdapter):
        def encode(self, text: Any, dimension: Any = None, instruct: Any = None) -> Any:
            if text == "hello":
                raise RuntimeError("provider dropped this one")
            return [0.5, 0.5]

    conn = _Conn({"embeddings_old": [_row(chunk_id="c1"), _row(chunk_id="c2")]})
    _wire(monkeypatch, conn, adapter=_Dropping())

    report = mod.embed_collections("new", use_async=True)

    assert report.failed is True
    assert "provider dropped" in report.collections[0].failures[0].error


def test_the_migration_writes_no_other_table(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn({"embeddings_old": [_row()]})
    writes = _wire(monkeypatch, conn)

    mod.embed_collections("new")

    assert set(conn.tables) == {"embeddings_old", "embeddings_new"}
    for call in writes:
        assert all(item.model == "new" for item in call["embeddings"])


def test_main_returns_0_on_a_clean_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire(monkeypatch, _Conn({"embeddings_old": [_row()]}))
    monkeypatch.setattr(
        mod,
        "embed_collections",
        lambda *args, **kwargs: mod.EmbedReport(
            model_id="new",
            dry_run=False,
            collections=[mod.CollectionReport(name="kb", documents_embedded=1)],
        ),
    )

    assert mod.main(["--model", "new", "--collection", "kb"]) == 0


def test_main_returns_1_when_a_document_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "embed_collections",
        lambda *args, **kwargs: mod.EmbedReport(
            model_id="new",
            dry_run=False,
            collections=[
                mod.CollectionReport(
                    name="kb", failures=[mod.DocumentFailure("d", "x")]
                )
            ],
        ),
    )

    assert mod.main(["--model", "new"]) == 1


def test_main_returns_2_when_the_run_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise MigrationCliError("not configured")

    monkeypatch.setattr(mod, "embed_collections", _raise)

    assert mod.main(["--model", "new"]) == 2


def test_main_rejects_bad_numbers() -> None:
    assert mod.main(["--model", "new", "--batch-size", "0"]) == 2
    assert mod.main(["--model", "new", "--rate-limit", "-1"]) == 2


def test_main_forwards_the_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        mod,
        "embed_collections",
        lambda model, **kwargs: seen.append({"model": model, **kwargs})
        or mod.EmbedReport(model_id=model, dry_run=False),
    )

    mod.main(
        [
            "--model",
            "new",
            "--collection",
            "a",
            "--dry-run",
            "--rate-limit",
            "5",
            "--use-async",
            "--batch-size",
            "7",
        ]
    )

    assert seen == [
        {
            "model": "new",
            "collections": ["a"],
            "team": None,
            "dry_run": True,
            "rate_limit": 5.0,
            "batch_size": 7,
            "use_async": True,
        }
    ]


def test_a_short_vector_batch_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Short(_StubAdapter):
        def encode(self, text: Any, dimension: Any = None, instruct: Any = None) -> Any:
            return [[0.5, 0.5]]

    conn = _Conn({"embeddings_old": [_row(chunk_id="c1"), _row(chunk_id="c2")]})
    _write_calls: List[Any] = []
    monkeypatch.setattr(mod, "connection", lambda: conn)
    monkeypatch.setattr(
        mod,
        "collection_info",
        lambda name: __import__("types").SimpleNamespace(embedding_model_id="old"),
    )
    monkeypatch.setattr(mod, "collection_engine", lambda info: "lancedb")
    monkeypatch.setattr(
        mod,
        "resolve_target_model",
        lambda model_id: (
            EmbeddingModelConfig(id=model_id, model_name="stub", dimension=2),
            _Short(),
        ),
    )
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["kb"])
    monkeypatch.setattr(
        mod, "write_vectors_to_db", lambda **kwargs: _write_calls.append(kwargs)
    )

    report = mod.embed_collections("new")

    assert _write_calls == []
    assert "provider returned 1 vectors" in report.collections[0].failures[0].error


def test_the_target_indexes_are_built_before_the_command_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, _Conn({"embeddings_old": [_row()]}))
    calls: List[Any] = []
    monkeypatch.setattr(
        mod,
        "_ensure_indexes",
        lambda name, model_id, report, **kwargs: calls.append((name, model_id)),
    )

    mod.embed_collections("new")

    assert calls == [("kb", "new")]


@pytest.mark.parametrize("status", ["failed", "index_corrupted"])
def test_a_broken_target_index_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import factory

    class _Store:
        def create_index(self, model_id: str) -> Any:
            return __import__("types").SimpleNamespace(
                status=status, advice="boom", fts_enabled=True
            )

    monkeypatch.setattr(factory, "get_vector_index_store", lambda: _Store())
    monkeypatch.setattr(mod, "embeddings_table", lambda model_id, conn: "t")
    report = mod.CollectionReport(name="kb", documents_unchanged=1)

    mod._ensure_indexes("kb", "new", report)

    assert status in report.failures[0].error


def test_an_unchanged_collection_needs_no_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import factory

    class _Store:
        def create_index(self, model_id: str) -> Any:
            return __import__("types").SimpleNamespace(
                status="below_threshold", advice=None, fts_enabled=False
            )

    monkeypatch.setattr(factory, "get_vector_index_store", lambda: _Store())
    report = mod.CollectionReport(name="kb", documents_unchanged=1)

    mod._ensure_indexes("kb", "new", report)

    assert report.failures == []


def test_a_target_without_a_full_text_index_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import factory

    class _Store:
        def create_index(self, model_id: str) -> Any:
            return __import__("types").SimpleNamespace(
                status="below_threshold", fts_enabled=False
            )

    monkeypatch.setattr(factory, "get_vector_index_store", lambda: _Store())
    monkeypatch.setattr(mod, "embeddings_table", lambda model_id, conn: "t")
    report = mod.CollectionReport(name="kb", documents_unchanged=0)

    mod._ensure_indexes("kb", "new", report)

    assert report.failures[0].doc_id == "*"


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([0.5], "2-dimension"),
        ([0.5, float("nan")], "non-finite"),
        ([0.0, 0.0], "all-zero"),
        ("nope", "non-list"),
    ],
)
def test_an_unusable_vector_is_rejected(vector: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        mod._validate_vector(vector, 2)


def test_main_returns_1_when_every_collection_was_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mod,
        "embed_collections",
        lambda *args, **kwargs: mod.EmbedReport(
            model_id="new",
            dry_run=False,
            collections=[
                mod.CollectionReport(name="kb", collection_skip="other-engine")
            ],
        ),
    )

    assert mod.main(["--model", "new"]) == 1


def test_main_rejects_a_non_finite_rate_limit() -> None:
    assert mod.main(["--model", "new", "--rate-limit", "nan"]) == 2


# --- real LanceDB: the copy must land in a table search can use ---


def _make_collection(name: str = "kb") -> None:
    import asyncio

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    asyncio.run(
        cm.collection_manager.save_collection(
            CollectionInfo(
                name=name, embedding_model_id="old-model", embedding_dimension=2
            )
        )
    )


def _config() -> Any:
    from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig

    return IngestionConfig(embedding_model_id="new-model")


class _FixedAdapter:
    def __init__(self, dimension: int = 2) -> None:
        self.dimension = dimension
        self.calls: List[List[str]] = []

    def encode(self, texts: Any) -> List[List[float]]:
        items = texts if isinstance(texts, list) else [texts]
        self.calls.append(list(items))
        return [[0.5, 0.5][: self.dimension] for _ in items]


def _install_source_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Four rows over two documents, readable one document at a time."""
    monkeypatch.setattr(
        mod, "iter_source_doc_ids", lambda *a, **k: (["doc-0", "doc-1"], 0)
    )
    monkeypatch.setattr(
        mod,
        "iter_source_chunks",
        lambda *a, doc_id=None, **k: iter(
            [
                chunk
                for chunk in _source_chunks(4)
                if doc_id is None or chunk.doc_id == doc_id
            ]
        ),
    )


def _source_chunks(count: int) -> List[Any]:
    from xagent.migrations.embedding.scope import SearchableChunk

    return [
        SearchableChunk(
            doc_id=f"doc-{index // 2}",
            chunk_id=f"chunk-{index}",
            parse_hash="h",
            text=f"text {index}",
            chunk_hash=f"ch-{index}",
            metadata=None,
            user_id=7,
        )
        for index in range(count)
    ]


def _indexes(model_id: str) -> List[Any]:
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        embeddings_table_candidates,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    conn = get_vector_index_store().get_raw_connection()
    table = conn.open_table(embeddings_table_candidates(model_id)[0])
    return table.list_indices()


def test_embed_writes_a_real_table_and_builds_both_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.core import config as core_config
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        embeddings_table_candidates,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    _make_collection()
    _install_source_stub(monkeypatch)
    # A real vector index instead of the 50k rows its default threshold needs.
    real_policy = core_config.IndexPolicy

    def _small_policy(*args: Any, **kwargs: Any) -> Any:
        return real_policy(enable_threshold_rows=1, *args, **kwargs)

    monkeypatch.setattr(core_config, "IndexPolicy", _small_policy)
    target = EmbeddingModelConfig(id="new-model", model_name="stub", dimension=2)

    report = mod.embed_collection(
        "kb",
        model_id="new-model",
        embedding_config=target,
        embedding_adapter=_FixedAdapter(),
        cfg=_config(),
        dry_run=False,
        source="old-model",
    )

    assert (report.documents_embedded, report.chunks_embedded) == (2, 4)
    assert report.failures == []
    conn = get_vector_index_store().get_raw_connection()
    table = conn.open_table(embeddings_table_candidates("new-model")[0])
    rows = table.search().where("collection = 'kb'").to_arrow().to_pylist()
    assert sorted(row["chunk_id"] for row in rows) == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert {row["text"] for row in rows} == {"text 0", "text 1", "text 2", "text 3"}
    assert {row["user_id"] for row in rows} == {7}

    indexes = _indexes("new-model")
    assert any(
        index.index_type == "FTS" and "text" in index.columns for index in indexes
    )
    assert any("vector" in index.columns for index in indexes)


def test_a_real_rerun_over_held_chunks_writes_nothing() -> None:
    _make_collection()
    monkeypatch = pytest.MonkeyPatch()
    _install_source_stub(monkeypatch)
    monkeypatch.setattr(mod, "_ensure_indexes", lambda *a, **k: None)
    target = EmbeddingModelConfig(id="new-model", model_name="stub", dimension=2)
    adapter = _FixedAdapter()

    def _run() -> Any:
        return mod.embed_collection(
            "kb",
            model_id="new-model",
            embedding_config=target,
            embedding_adapter=adapter,
            cfg=_config(),
            dry_run=False,
            source="old-model",
        )

    try:
        first = _run()
        calls = len(adapter.calls)
        second = _run()
    finally:
        monkeypatch.undo()

    assert (first.chunks_embedded, second.chunks_embedded) == (4, 0)
    assert second.documents_unchanged == first.documents_embedded
    assert len(adapter.calls) == calls


# --- round 1: one physical table for the held scan and the writer ---


def _legacy_name(model_id: str) -> Any:
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        legacy_embeddings_table_name,
    )

    return legacy_embeddings_table_name(model_id)


def _create_embeddings_table(conn: Any, name: str, rows: Any) -> None:
    schema = pa.schema(
        [
            ("collection", pa.string()),
            ("doc_id", pa.string()),
            ("chunk_id", pa.string()),
            ("parse_hash", pa.string()),
            ("user_id", pa.int64()),
            ("text", pa.string()),
            ("chunk_hash", pa.string()),
            ("metadata", pa.string()),
            ("vector", pa.list_(pa.float32(), 2)),
        ]
    )
    conn.create_table(name, data=pa.Table.from_pylist(rows, schema=schema))


def _embedding_row(chunk_id: str, model_id: str) -> Any:
    return {
        "collection": "kb",
        "doc_id": "doc-0",
        "chunk_id": chunk_id,
        "parse_hash": "h",
        "user_id": 7,
        "text": "t",
        "chunk_hash": "c",
        "metadata": None,
        "vector": [0.5, 0.5],
    }


def _stub_legacy_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    monkeypatch.setattr(
        model_resolver,
        "resolve_embedding_adapter",
        lambda model_id: (SimpleNamespace(model_name="stub"), object()),
    )


def test_a_populated_legacy_table_is_refused_instead_of_shadowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    _make_collection()
    _install_source_stub(monkeypatch)
    _stub_legacy_model(monkeypatch)
    monkeypatch.setattr(
        mod,
        "resolve_target_model",
        lambda model_id: (
            EmbeddingModelConfig(id=model_id, model_name="stub", dimension=2),
            _FixedAdapter(),
        ),
    )
    monkeypatch.setattr(mod, "_ensure_indexes", lambda *a, **k: None)
    conn = get_vector_index_store().get_raw_connection()
    legacy = _legacy_name("new-model")
    _create_embeddings_table(conn, legacy, [_embedding_row("legacy-1", "new-model")])

    report = mod.embed_collection(
        "kb",
        model_id="new-model",
        embedding_config=EmbeddingModelConfig(
            id="new-model", model_name="stub", dimension=2
        ),
        embedding_adapter=_FixedAdapter(),
        cfg=_config(),
        dry_run=False,
        source="old-model",
    )

    assert report.collection_skip == mod.SKIP_LEGACY_SHADOWED
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        embeddings_table_candidates,
    )

    assert embeddings_table_candidates("new-model")[0] not in set(
        conn.list_tables() if isinstance(conn.list_tables(), list) else []
    )


def test_held_rows_are_read_from_the_table_search_would_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    _stub_legacy_model(monkeypatch)

    conn = get_vector_index_store().get_raw_connection()
    legacy = _legacy_name("new-model")
    _create_embeddings_table(conn, legacy, [_embedding_row("legacy-1", "new-model")])

    held = mod.target_chunk_ids("kb", "new-model", dimension=2)
    mod.target_chunk_ids("kb", "new-model", dimension=2)

    from xagent.migrations.embedding import scope as scope_mod

    assert held == {"legacy-1"}
    assert scope_mod.source_embeddings_table("new-model", conn) == legacy


def test_a_populated_target_without_fts_fails_even_when_nothing_was_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import factory

    class _Store:
        def create_index(self, model_id: str) -> Any:
            return __import__("types").SimpleNamespace(
                status="below_threshold", advice=None, fts_enabled=False
            )

    monkeypatch.setattr(factory, "get_vector_index_store", lambda: _Store())
    monkeypatch.setattr(mod, "embeddings_table", lambda model_id, conn: "t")
    report = mod.CollectionReport(name="kb", documents_unchanged=3)

    mod._ensure_indexes("kb", "new", report)

    assert report.failures[0].doc_id == "*"


def test_a_collection_with_no_target_table_needs_no_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.storage import factory

    class _Store:
        def create_index(self, model_id: str) -> Any:  # pragma: no cover
            raise AssertionError("no table, no index call")

    monkeypatch.setattr(factory, "get_vector_index_store", lambda: _Store())
    monkeypatch.setattr(mod, "embeddings_table", lambda model_id, conn: None)
    report = mod.CollectionReport(name="kb")

    mod._ensure_indexes("kb", "new", report)

    assert report.failures == []


def test_a_document_is_settled_once_across_many_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.migrations.embedding.scope import SearchableChunk

    many = [
        SearchableChunk(
            doc_id="doc-0",
            chunk_id=f"c{index}",
            parse_hash="h",
            text="t",
            chunk_hash="c",
            metadata=None,
            user_id=7,
        )
        for index in range(120)
    ]
    _wire(monkeypatch, _Conn({"embeddings_old": []}))
    monkeypatch.setattr(mod, "iter_source_doc_ids", lambda *a, **k: (["doc-0"], 0))
    monkeypatch.setattr(mod, "iter_source_chunks", lambda *a, **k: iter(many))
    written: List[int] = []
    real_write = mod.embed_collection.__globals__["_write_chunks"]

    def _record(*args: Any, **kwargs: Any) -> None:
        written.append(len(args[1]))
        real_write(*args, **kwargs)

    monkeypatch.setattr(mod, "_write_chunks", _record)

    report = mod.embed_collection(
        "kb",
        model_id="new",
        embedding_config=EmbeddingModelConfig(id="new", model_name="stub", dimension=2),
        embedding_adapter=_FixedAdapter(),
        cfg=_config(),
        dry_run=False,
        source="old",
    )

    assert written == [120]
    assert (report.documents_embedded, report.documents_total) == (1, 1)
