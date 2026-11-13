"""Tests for the embedding migration's read-only scope."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import pyarrow as pa
import pytest

from xagent.migrations.embedding import scope as mod
from xagent.migrations.embedding.scope import MigrationCliError, SearchableChunk


@pytest.fixture(autouse=True)
def _stub_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """--model resolution needs the model hub; tests pin it to a 2-d stub."""
    monkeypatch.setattr(mod, "resolve_target", lambda model_id: (model_id, 2))


class _Table:
    def __init__(self, rows: List[Dict[str, Any]], *, fail: bool = False) -> None:
        self._rows = rows
        self._fail = fail
        self.filters: List[str] = []
        names = (
            list(rows[0])
            if rows
            else ["doc_id", "chunk_id", "user_id", "text", "vector"]
        )
        self.schema = pa.schema([(name, pa.string()) for name in names])

    def to_batches(self, *, filter: str, batch_size: int) -> Iterator[Any]:
        self.filters.append(filter)
        if self._fail:
            raise RuntimeError("corrupt table")
        for start in range(0, max(len(self._rows), 1), 2):
            rows = self._rows[start : start + 2]
            if rows:
                yield pa.RecordBatch.from_pylist(rows)

    def search(self) -> "_Query":
        return _Query(self)

    def close(self) -> None:  # pragma: no cover - close is best-effort
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
    def __init__(self, tables: Dict[str, _Table]) -> None:
        self.tables = tables
        self.created: List[str] = []

    def open_table(self, name: str) -> _Table:
        if name not in self.tables:
            raise ValueError(f"Table '{name}' was not found")
        return self.tables[name]

    def create_table(self, name: str, **kwargs: Any) -> None:
        self.created.append(name)

    def list_tables(self) -> List[str]:
        return list(self.tables)


def _conn(tables: Dict[str, _Table]) -> _Conn:
    return _Conn(tables)


def _row(**overrides: Any) -> Dict[str, Any]:
    row = {
        "doc_id": "doc-1",
        "chunk_id": "chunk-1",
        "parse_hash": "hash-1",
        "text": "hello",
        "chunk_hash": "ch-1",
        "metadata": None,
        "user_id": 7,
        "vector": [0.5, 0.5],
    }
    row.update(overrides)
    return row


def test_rows_are_streamed_in_batches() -> None:
    table = _Table([_row(chunk_id=f"c{index}") for index in range(5)])
    conn = _conn({"embeddings_new": table})

    rows = list(
        mod.iter_rows(conn, "embeddings_new", "collection = 'kb'", ["chunk_id"])
    )

    assert [row["chunk_id"] for row in rows] == ["c0", "c1", "c2", "c3", "c4"]
    assert table.filters == ["collection = 'kb'"]


def test_a_failed_read_raises_instead_of_looking_empty() -> None:
    conn = _conn({"embeddings_old": _Table([], fail=True)})

    with pytest.raises(RuntimeError, match="corrupt table"):
        list(mod.iter_source_chunks("kb", "old", conn=conn))


def test_a_missing_table_is_an_empty_collection() -> None:
    conn = _conn({})

    assert list(mod.iter_source_chunks("kb", "old", conn=conn)) == []


def test_the_chunk_carries_everything_a_copy_needs() -> None:
    conn = _conn({"embeddings_old": _Table([_row()])})

    chunk = next(iter(mod.iter_source_chunks("kb", "old", conn=conn)))

    assert chunk == SearchableChunk(
        doc_id="doc-1",
        chunk_id="chunk-1",
        parse_hash="hash-1",
        text="hello",
        chunk_hash="ch-1",
        metadata=None,
        user_id=7,
    )
    assert mod.skip_reason(chunk) is None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"user_id": None}, mod.SKIP_NO_OWNER),
        ({"doc_id": ""}, mod.SKIP_NO_DOC_ID),
        ({"chunk_id": ""}, mod.SKIP_NO_DOC_ID),
        ({"text": ""}, mod.SKIP_NO_TEXT),
    ],
)
def test_each_unusable_row_has_exactly_one_reason(
    overrides: Dict[str, Any], reason: str
) -> None:
    conn = _conn({"embeddings_old": _Table([_row(**overrides)])})

    chunk = next(iter(mod.iter_source_chunks("kb", "old", conn=conn)))

    assert mod.skip_reason(chunk) == reason


def test_target_chunks_default_to_empty_when_the_table_does_not_exist() -> None:
    assert mod.target_chunk_ids("kb", "new", conn=_conn({})) == set()


def test_target_chunks_are_strict_too() -> None:
    conn = _conn({"embeddings_new": _Table([], fail=True)})

    with pytest.raises(RuntimeError):
        mod.target_chunk_ids("kb", "new", conn=conn)


def _wire_collection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    engine: str = "lancedb",
    bound: Optional[str] = "old",
    info_error: Optional[BaseException] = None,
) -> None:
    def _info(name: str) -> Any:
        if info_error is not None:
            raise info_error
        return SimpleNamespace(embedding_model_id=bound)

    monkeypatch.setattr(mod, "collection_info", _info)
    monkeypatch.setattr(mod, "collection_engine", lambda info: engine)


def test_the_report_counts_documents_chunks_and_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire_collection(monkeypatch)
    conn = _conn(
        {
            "embeddings_old": _Table(
                [
                    _row(doc_id="a", chunk_id="a1"),
                    _row(doc_id="a", chunk_id="a2"),
                    _row(doc_id="b", chunk_id="b1"),
                    _row(doc_id="legacy", chunk_id="l1", user_id=None),
                    _row(doc_id="empty", chunk_id="e1", text=""),
                ]
            )
        }
    )

    item = mod.scope_collection("kb", "new", conn=conn)

    assert item.engine == "lancedb"
    assert item.source_model_id == "old"
    assert item.documents == 2
    assert item.chunks == 5
    assert item.skipped == {mod.SKIP_NO_OWNER: 1, mod.SKIP_NO_TEXT: 1}
    assert item.usable is True


def test_the_missing_count_uses_the_target_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire_collection(monkeypatch)
    conn = _conn(
        {
            "embeddings_old": _Table(
                [_row(chunk_id="a1"), _row(chunk_id="a2"), _row(chunk_id="a3")]
            ),
            "embeddings_new": _Table([{"chunk_id": "a1", "vector": [0.5, 0.5]}]),
        }
    )

    item = mod.scope_collection("kb", "new", conn=conn)

    assert item.chunks == 3
    assert item.missing_chunks == 2


def test_without_a_target_model_nothing_is_counted_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire_collection(monkeypatch)

    item = mod.scope_collection("kb", conn=_conn({"embeddings_old": _Table([_row()])}))

    assert item.missing_chunks == 0


@pytest.mark.parametrize(
    ("engine", "bound", "reason"),
    [
        ("milvus", "old", mod.SKIP_OTHER_ENGINE),
        ("", "old", mod.SKIP_UNREADABLE_BINDING),
        ("lancedb", None, mod.SKIP_NO_SOURCE_MODEL),
    ],
)
def test_a_collection_that_cannot_be_migrated_is_reported_not_read(
    monkeypatch: pytest.MonkeyPatch, engine: str, bound: Optional[str], reason: str
) -> None:
    _wire_collection(monkeypatch, engine=engine, bound=bound)
    table = _Table([_row()])
    conn = _conn({"embeddings_old": table})

    item = mod.scope_collection("kb", "new", conn=conn)

    assert item.skip_reason == reason
    assert item.chunks == 0
    assert table.filters == []


def test_an_unknown_collection_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire_collection(
        monkeypatch, info_error=MigrationCliError("collection 'kb' does not exist")
    )

    with pytest.raises(MigrationCliError, match="does not exist"):
        mod.scope_collection("kb", "new", conn=_conn({}))


def test_an_unreadable_binding_does_not_stop_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.kb import coordinator as coordinator_mod

    def _explode(info: object | None) -> Any:
        raise ValueError("garbage backend")

    monkeypatch.setattr(coordinator_mod, "resolve_collection_backend", _explode)

    assert mod.collection_engine(SimpleNamespace()) == ""


def test_collection_info_reports_a_missing_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import (
        collection_manager as collection_manager_mod,
    )

    def _missing(name: str) -> Any:
        raise ValueError("Collection not found")

    monkeypatch.setattr(collection_manager_mod, "get_collection_sync", _missing)

    with pytest.raises(MigrationCliError, match="does not exist"):
        mod.collection_info("kb")


def test_collection_info_does_not_hide_other_read_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import (
        collection_manager as collection_manager_mod,
    )

    def _broken(name: str) -> Any:
        raise OSError("disk gone")

    monkeypatch.setattr(collection_manager_mod, "get_collection_sync", _broken)

    with pytest.raises(OSError):
        mod.collection_info("kb")


def test_explicit_collections_are_trimmed_and_deduplicated() -> None:
    assert mod.resolve_collection_names(collections=[" kb ", "kb", "other"]) == [
        "kb",
        "other",
    ]


def test_collection_and_team_are_mutually_exclusive() -> None:
    with pytest.raises(MigrationCliError, match="not both"):
        mod.resolve_collection_names(collections=["kb"], team=3)


def test_an_empty_collection_name_is_rejected() -> None:
    with pytest.raises(MigrationCliError, match="must not be empty"):
        mod.resolve_collection_names(collections=["  "])


def test_every_collection_is_read_from_the_metadata_table_without_creating_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _conn(
        {mod.METADATA_TABLE: _Table([{"name": "b"}, {"name": "a"}, {"name": ""}])}
    )
    monkeypatch.setattr(mod, "connection", lambda: conn)

    assert mod.resolve_collection_names() == ["a", "b"]
    assert conn.created == []


def test_a_missing_metadata_table_lists_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "connection", lambda: _conn({}))

    assert mod.resolve_collection_names() == []


def test_a_failed_metadata_read_is_not_reported_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _conn({mod.METADATA_TABLE: _Table([{"name": "a"}], fail=True)})
    monkeypatch.setattr(mod, "connection", lambda: conn)

    with pytest.raises(RuntimeError, match="corrupt table"):
        mod.resolve_collection_names()


def test_team_scope_requires_the_application_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services import knowledge_base_team_scope as team_scope

    monkeypatch.setattr(team_scope, "team_knowledge_base_hook_installed", lambda: False)

    with pytest.raises(MigrationCliError, match="--collection"):
        mod.resolve_collection_names(team=7)


def test_team_scope_maps_hook_answers_to_collection_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.models import database
    from xagent.web.services import knowledge_base_team_scope as team_scope
    from xagent.web.services.knowledge_base_team_scope import KnowledgeBaseAccess

    seen: Dict[str, Any] = {}

    def _configure_db(*args: Any, **kwargs: Any) -> None:
        seen["configure_db"] = kwargs

    class _Session:
        def close(self) -> None:
            seen["closed"] = True

    def _team_knowledge_bases(
        session: Any, *, team_id: int
    ) -> List[KnowledgeBaseAccess]:
        seen["team_id"] = team_id
        return [
            KnowledgeBaseAccess(name="b", storage_user_id=11),
            KnowledgeBaseAccess(name="a", storage_user_id=12),
        ]

    monkeypatch.setattr(team_scope, "team_knowledge_base_hook_installed", lambda: True)
    monkeypatch.setattr(database, "configure_db", _configure_db)
    monkeypatch.setattr(database, "get_session_local", lambda: _Session)
    monkeypatch.setattr(team_scope, "team_knowledge_bases", _team_knowledge_bases)

    assert mod.resolve_collection_names(team=7) == ["a", "b"]
    assert seen["team_id"] == 7
    assert seen["configure_db"] == {"read_only": True}
    assert seen["closed"] is True


def test_the_report_never_creates_a_table(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire_collection(monkeypatch)
    conn = _conn({"embeddings_old": _Table([_row()])})
    monkeypatch.setattr(mod, "connection", lambda: conn)
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["kb"])

    mod.scope_report(target_model_id="new")

    assert conn.created == []


def test_scope_report_reads_with_one_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire_collection(monkeypatch)
    conn = _conn({"embeddings_old": _Table([_row()])})
    opened: List[int] = []

    def _connection() -> Any:
        opened.append(1)
        return conn

    monkeypatch.setattr(mod, "connection", _connection)
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["a", "b"])

    report = mod.scope_report(target_model_id="new")

    assert [item.name for item in report] == ["a", "b"]
    assert len(opened) == 1


def test_main_prints_each_collection_and_returns_0(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        mod,
        "scope_report",
        lambda **kwargs: [
            mod.CollectionScope(
                name="kb",
                engine="lancedb",
                source_model_id="old",
                documents=2,
                chunks=3,
                missing_chunks=1,
                skipped={mod.SKIP_NO_OWNER: 1},
            )
        ],
    )

    with caplog.at_level("INFO", logger=mod.__name__):
        assert mod.main(["--model", "new"]) == 0

    assert "kb: source model old; 2 documents; 3 chunks; 1 missing" in caplog.text


def test_main_forwards_the_scope_and_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: List[Dict[str, Any]] = []
    monkeypatch.setattr(mod, "scope_report", lambda **kwargs: seen.append(kwargs) or [])

    mod.main(["--model", "new", "--collection", "a", "--collection", "b"])
    mod.main(["--team", "4"])

    assert seen == [
        {"collections": ["a", "b"], "team": None, "target_model_id": "new"},
        {"collections": None, "team": 4, "target_model_id": None},
    ]


def test_main_returns_1_when_no_collection_is_in_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "scope_report", lambda **kwargs: [])

    assert mod.main([]) == 1


def test_main_returns_2_when_the_scope_cannot_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(**kwargs: Any) -> Any:
        raise MigrationCliError("no hook")

    monkeypatch.setattr(mod, "scope_report", _fail)

    assert mod.main(["--team", "3"]) == 2


def test_main_returns_2_when_a_read_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(**kwargs: Any) -> Any:
        raise OSError("corrupt table")

    monkeypatch.setattr(mod, "scope_report", _fail)

    assert mod.main(["--collection", "kb"]) == 2


def test_main_rejects_collection_with_team() -> None:
    with pytest.raises(SystemExit) as exc:
        mod.main(["--collection", "a", "--team", "4"])
    assert exc.value.code == 2


def test_the_report_creates_no_table_and_writes_nothing_beyond_opening_the_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    from pathlib import Path

    root = Path(os.environ["LANCEDB_DIR"])
    # Opening a LanceDB connection writes its own manifest; the report must not
    # add a table or any other file on top of that.
    mod.connection()
    before = {path for path in root.rglob("*") if "__manifest" not in path.parts}

    with pytest.raises(MigrationCliError):
        mod.scope_report(collections=["kb"])

    after = {path for path in root.rglob("*") if "__manifest" not in path.parts}
    assert after == before
    assert not list(root.glob("*.lance"))


def test_a_target_row_without_a_vector_is_still_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire_collection(monkeypatch)
    conn = _conn(
        {
            "embeddings_old": _Table([_row(chunk_id="a1")]),
            "embeddings_new": _Table([{"chunk_id": "a1", "vector": None}]),
        }
    )

    item = mod.scope_collection("kb", "new", conn=conn)

    assert item.missing_chunks == 1


def test_a_legacy_named_table_is_read_like_search_reads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mod,
        "_searchable_tables",
        lambda model_id: ["embeddings_x", "embeddings_legacy"],
    )
    conn = _conn({"embeddings_legacy": _Table([_row()])})

    chunks = list(mod.iter_source_chunks("kb", "old", conn=conn))

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1"]


def test_the_source_model_falls_back_to_the_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import (
        collection_manager as collection_manager_mod,
    )

    calls: List[Any] = []

    def _resolve(name: str, config_model_id: Optional[str] = None) -> str:
        calls.append((name, config_model_id))
        return "from-config"

    monkeypatch.setattr(
        collection_manager_mod, "resolve_effective_embedding_model_read_only", _resolve
    )

    assert mod.source_model_id("kb", SimpleNamespace(embedding_model_id=None)) == (
        "from-config"
    )
    assert calls == [("kb", None)]


@pytest.mark.parametrize("missing", ["text", "user_id", "doc_id", "chunk_id"])
def test_a_missing_required_column_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    row = {"doc_id": "d", "chunk_id": "c", "user_id": 1, "text": "t"}
    row.pop(missing)
    table = _Table([row])
    conn = _conn({"embeddings_old": table})

    with pytest.raises(MigrationCliError, match=f"no {missing} column"):
        list(mod.iter_source_chunks("kb", "old", conn=conn))


# --- real LanceDB: the reader must work on the store's own table API ---


def _real_table(conn: Any, name: str, rows: List[Dict[str, Any]]) -> None:
    conn.create_table(name, data=pa.Table.from_pylist(rows))


def _chunk_row(**overrides: Any) -> Dict[str, Any]:
    return {
        "collection": "kb",
        "doc_id": "doc-1",
        "chunk_id": "chunk-1",
        "parse_hash": "hash-1",
        "text": "hello",
        "chunk_hash": "ch-1",
        "metadata": None,
        "user_id": 7,
        "vector": [0.5, 0.5],
        **overrides,
    }


def _model(model_id: str, model_name: str, dimension: int = 2) -> Any:
    return SimpleNamespace(id=model_id, model_name=model_name, dimension=dimension)


def test_iter_source_chunks_reads_a_real_table() -> None:
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        embeddings_table_candidates,
    )

    conn = mod.connection()
    table = embeddings_table_candidates("old-model")[0]
    _real_table(
        conn,
        table,
        [
            _chunk_row(),
            _chunk_row(chunk_id="chunk-2", collection="other"),
            _chunk_row(chunk_id="chunk-3", text=None),
        ],
    )

    chunks = list(mod.iter_source_chunks("kb", "old-model", conn=conn))

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1", "chunk-3"]
    assert chunks[0].doc_id == "doc-1"
    assert chunks[0].chunk_hash == "ch-1"
    assert chunks[0].user_id == 7
    assert chunks[1].text == ""


def test_scope_reads_the_legacy_table_search_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    model = _model("old-model", "legacy name")
    monkeypatch.setattr(
        model_resolver, "resolve_embedding_adapter", lambda model_id: (model, object())
    )
    legacy = f"embeddings_{to_model_tag('legacy name')}"
    store = get_vector_index_store()
    conn = store.get_raw_connection()
    _real_table(conn, legacy, [_chunk_row(), _chunk_row(chunk_id="chunk-2")])

    _, search_table = store.open_embeddings_table("old-model")

    assert search_table == legacy
    assert mod.source_embeddings_table("old-model", conn) == search_table
    assert len(list(mod.iter_source_chunks("kb", "old-model", conn=conn))) == 2


def test_a_config_only_binding_resolves_like_search() -> None:
    import asyncio

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionInfo,
        IngestionConfig,
    )
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        get_collection_sync,
        resolve_effective_embedding_model_sync,
    )

    asyncio.run(
        cm.collection_manager.save_collection(
            CollectionInfo(
                name="kb",
                ingestion_config=IngestionConfig(embedding_model_id="old-model"),
            )
        )
    )

    assert mod.source_model_id("kb", get_collection_sync("kb")) == "old-model"
    assert resolve_effective_embedding_model_sync("kb") == "old-model"


def test_a_tool_layer_none_binding_is_not_a_model() -> None:
    import asyncio

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        get_collection_sync,
    )

    asyncio.run(
        cm.collection_manager.save_collection(
            CollectionInfo(name="kb", embedding_model_id="none")
        )
    )

    assert mod.source_model_id("kb", get_collection_sync("kb")) is None


@pytest.mark.parametrize("missing", ["doc_id", "chunk_id", "user_id", "text"])
def test_a_real_table_missing_a_required_column_is_a_hard_failure(missing: str) -> None:
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        embeddings_table_candidates,
    )

    conn = mod.connection()
    row = {"doc_id": "d", "chunk_id": "c", "user_id": 1, "text": "t"}
    row.pop(missing)
    _real_table(conn, embeddings_table_candidates("old-model")[0], [row])

    with pytest.raises(MigrationCliError, match=f"no {missing} column"):
        list(mod.iter_source_chunks("kb", "old-model", conn=conn))


def _metadata_snapshot() -> Any:
    """Every column of every collection_metadata row, attribute order included."""
    from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store

    conn = get_metadata_store().get_raw_connection()
    table = conn.open_table(mod.METADATA_TABLE)
    rows = table.search().where("true").to_arrow().to_pylist()
    return sorted(
        (r.get("name"), tuple(sorted((k, repr(v)) for k, v in r.items()))) for r in rows
    )


def test_scope_and_dry_run_write_nothing_to_collection_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    # An existing, uninitialized collection that has embeddings: the search
    # resolver would infer and persist a binding for it.
    asyncio.run(
        cm.collection_manager.save_collection(
            CollectionInfo(name="kb", documents=1, embeddings=2)
        )
    )
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        embeddings_table_candidates,
    )

    conn = mod.connection()
    _real_table(
        conn,
        embeddings_table_candidates("old-model")[0],
        [_chunk_row(), _chunk_row(chunk_id="chunk-2")],
    )
    monkeypatch.setattr(
        mod, "collection_info", lambda name: cm.get_collection_sync(name)
    )
    before = _metadata_snapshot()

    mod.scope_report(collections=["kb"])
    mod.source_model_id("kb", cm.get_collection_sync("kb"))
    mod.target_chunk_ids("kb", "new-model", conn=conn)

    assert _metadata_snapshot() == before


def test_an_explicit_model_is_resolved_like_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(mod, "resolve_target", lambda model_id: ("canonical-id", 3))
    monkeypatch.setattr(
        mod,
        "target_chunk_ids",
        lambda name, model_id, **kwargs: seen.update(
            {"name": name, "model": model_id, **kwargs}
        )
        or set(),
    )
    monkeypatch.setattr(mod, "collection_info", lambda name: SimpleNamespace())
    monkeypatch.setattr(mod, "collection_engine", lambda info: "lancedb")
    monkeypatch.setattr(mod, "source_model_id", lambda name, info: "source-model")
    monkeypatch.setattr(mod, "iter_source_chunks", lambda *a, **k: iter([]))

    mod.scope_collection("kb", "an-alias")

    assert seen["model"] == "canonical-id"
    assert seen["dimension"] == 3


def test_a_discovered_collection_with_a_schema_error_fails_the_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mod, "resolve_collection_names", lambda **kwargs: ["bad", "gone", "ok"]
    )

    def _scope(name: str, target: Any, *, conn: Any) -> Any:
        if name == "bad":
            raise mod.MigrationCliError("source table is missing column 'user_id'")
        if name == "gone":
            raise mod.MissingCollectionError("collection 'gone' does not exist")
        return mod.CollectionScope(name=name)

    monkeypatch.setattr(mod, "scope_collection", _scope)
    monkeypatch.setattr(mod, "connection", lambda: object())

    report = {item.name: item for item in mod.scope_report()}

    assert "user_id" in (report["bad"].error or "")
    assert not report["bad"].usable
    assert report["gone"].skip_reason == mod.SKIP_NO_METADATA
    assert report["gone"].error is None
    assert report["ok"].usable
    assert mod.main([]) == 1


def test_a_collection_that_vanished_after_discovery_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mod, "resolve_collection_names", lambda **kwargs: ["gone", "ok"]
    )

    def _scope(name: str, target: Any, *, conn: Any) -> Any:
        if name == "gone":
            raise mod.MissingCollectionError("collection 'gone' does not exist")
        return mod.CollectionScope(name=name)

    monkeypatch.setattr(mod, "scope_collection", _scope)
    monkeypatch.setattr(mod, "connection", lambda: object())

    assert mod.main([]) == 0


def test_a_missing_collection_raises_the_missing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    def _missing(name: str) -> Any:
        raise ValueError(f"Collection '{name}' not found")

    monkeypatch.setattr(cm, "get_collection_sync", _missing)

    with pytest.raises(mod.MissingCollectionError):
        mod.collection_info("kb")
