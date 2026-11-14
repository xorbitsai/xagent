"""Scope: eligibility, strict reads and per-collection isolation."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import pyarrow as pa
import pytest

from xagent.migrations.embedding import scope as mod

SCHEMA = pa.schema(
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


def _row(
    chunk_id: str = "c1",
    *,
    doc_id: Optional[str] = "d1",
    user_id: Optional[int] = 7,
    text: Optional[str] = "t",
    collection: str = "kb",
) -> Dict[str, Any]:
    return {
        "collection": collection,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "parse_hash": "p",
        "user_id": user_id,
        "text": text,
        "chunk_hash": "h",
        "metadata": None,
        "vector": [0.5, 0.5],
    }


class _Query:
    def __init__(self, table: "_Table") -> None:
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
        rows = [row for row in self.table.rows if _matches(row, self.filter)]
        for start in range(0, len(rows), batch_size):
            yield pa.RecordBatch.from_pylist(
                [
                    {key: row.get(key) for key in self.columns or row}
                    for row in rows[start : start + batch_size]
                ]
            )

    def to_arrow(self) -> Any:
        return pa.Table.from_pylist(
            [row for row in self.table.rows if _matches(row, self.filter)]
        )


def _matches(row: Dict[str, Any], where: str) -> bool:
    if not where or where == "true":
        return True
    for clause in where.split(" AND "):
        column, _, literal = clause.partition(" = ")
        literal = literal.strip("'")
        if str(row.get(column.strip())) != literal:
            return False
    return True


class _Table:
    def __init__(self, rows: List[Dict[str, Any]], names: Optional[List[str]] = None):
        self.rows = rows
        self.schema = pa.schema(
            [(name, pa.string()) for name in (names or list(SCHEMA.names))]
        )

    def search(self) -> _Query:
        return _Query(self)

    def count_rows(self, filter: Optional[str] = None) -> int:
        return len([row for row in self.rows if _matches(row, filter or "")])


class _Conn:
    def __init__(self, tables: Dict[str, _Table]) -> None:
        self.tables = tables

    def open_table(self, name: str) -> _Table:
        if name not in self.tables:
            raise FileNotFoundError(name)
        return self.tables[name]

    def list_tables(self) -> List[str]:
        return list(self.tables)


def _bound(
    model_id: Optional[str] = "BAAI/old-model",
    dimension: Optional[int] = 2,
    name: str = "kb",
) -> Dict[str, Any]:
    return {
        "name": name,
        "embedding_model_id": model_id,
        "embedding_dimension": dimension,
        "extra_metadata": "{}",
    }


def _metadata(rows: List[Dict[str, Any]]) -> _Table:
    return _Table(rows, names=["name", "embedding_model_id", "embedding_dimension"])


def _wire(
    monkeypatch: pytest.MonkeyPatch, conn: _Conn, engine: str = "lancedb"
) -> None:
    monkeypatch.setattr(mod, "connection", lambda: conn)
    monkeypatch.setattr(mod, "collection_engine", lambda info: engine)
    monkeypatch.setattr(mod, "_legacy_spelling", lambda model_id: None)


def _writer(model_id: str = "BAAI/old-model") -> str:
    return mod.source_table(model_id)


# --- eligibility ---


def test_an_eligible_collection_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row("c1"), _row("c2")]),
        }
    )
    _wire(monkeypatch, conn)

    item = mod.scope_collection("kb")

    assert item.usable
    assert item.source_model_id == "BAAI/old-model"
    assert (item.documents, item.chunks) == (1, 2)
    assert mod.describe(item).startswith("kb: source model BAAI/old-model")


def test_an_eligible_collection_without_rows_is_counted_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()])})
    _wire(monkeypatch, conn)

    item = mod.scope_collection("kb")

    assert item.usable and (item.documents, item.chunks) == (0, 0)
    assert mod.source_eligibility("kb").table == _writer()


def test_an_unbound_collection_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound(None, None)]),
            _writer(): _Table([_row()]),
        }
    )
    _wire(monkeypatch, conn)

    assert mod.source_eligibility("kb").reason == mod.SKIP_UNBOUND
    assert mod.scope_collection("kb").skip_reason == mod.SKIP_UNBOUND


def test_a_binding_without_a_dimension_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound("BAAI/old-model", None)]),
            _writer(): _Table([_row()]),
        }
    )
    _wire(monkeypatch, conn)

    assert mod.source_eligibility("kb").reason == mod.SKIP_BOUND_WITHOUT_DIMENSION


def test_rows_in_another_table_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            "embeddings_BAAI_old_model": _Table([_row()]),
        }
    )
    _wire(monkeypatch, conn)

    assert _writer() == "embeddings_baai_old_model"
    assert mod.source_eligibility("kb").reason == mod.SKIP_ROWS_IN_OTHER_TABLE


def test_rows_split_across_tables_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row()]),
            "embeddings_BAAI_old_model": _Table([_row("c2")]),
        }
    )
    _wire(monkeypatch, conn)

    assert mod.source_eligibility("kb").reason == mod.SKIP_ROWS_IN_MULTIPLE_TABLES


def test_rows_of_another_collection_do_not_disqualify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row()]),
            "embeddings_legacy_name": _Table([_row(collection="other")]),
        }
    )
    _wire(monkeypatch, conn)

    assert mod.source_eligibility("kb").reason is None


def test_a_missing_required_column_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _Table([_row()])
    table.schema = pa.schema([(name, pa.string()) for name in ("doc_id", "user_id")])
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()]), _writer(): table})
    _wire(monkeypatch, conn)

    assert mod.source_eligibility("kb").reason == mod.SKIP_MISSING_COLUMN


def test_another_engine_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()])})
    _wire(monkeypatch, conn, engine="milvus")

    assert mod.source_eligibility("kb").reason == mod.SKIP_OTHER_ENGINE


def test_an_unreadable_engine_binding_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()])})
    _wire(monkeypatch, conn, engine="")

    assert mod.source_eligibility("kb").reason == mod.SKIP_UNREADABLE_BINDING


# --- counts ---


def test_excluded_rows_are_counted_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table(
                [
                    _row("c1"),
                    _row("c2", doc_id=None),
                    _row("c3", user_id=None),
                    _row("c4", text=None),
                    # A chunkless row beside a usable sibling of its document:
                    # it must be counted at the document read, not twice.
                    _row("", doc_id="d9"),
                    _row("c5", doc_id="d9"),
                ]
            ),
        }
    )
    _wire(monkeypatch, conn)

    item = mod.scope_collection("kb")

    assert item.chunks == 6
    assert item.skipped == {
        mod.SKIP_NO_DOC_ID: 2,
        mod.SKIP_NO_OWNER: 1,
        mod.SKIP_NO_TEXT: 1,
    }
    assert item.documents == 2


def test_the_missing_count_uses_the_target_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row("c1"), _row("c2"), _row("c3")]),
            mod.source_table("new-model"): _Table([_row("c1")]),
        }
    )
    _wire(monkeypatch, conn)
    monkeypatch.setattr(mod, "resolve_target", lambda model_id: ("new-model", 2))

    assert mod.scope_collection("kb", "new-model").missing_chunks == 2


def test_a_target_row_with_a_bad_vector_is_still_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = _row("c1")
    bad["vector"] = [0.0, 0.0]
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row("c1")]),
            mod.source_table("new-model"): _Table([bad]),
        }
    )
    _wire(monkeypatch, conn)
    monkeypatch.setattr(mod, "resolve_target", lambda model_id: ("new-model", 2))

    assert mod.scope_collection("kb", "new-model").missing_chunks == 1


# --- strict reads and isolation ---


def test_an_absent_row_is_the_only_missing_collection() -> None:
    with pytest.raises(mod.MissingCollectionError):
        mod.collection_info("kb")


def test_a_missing_collection_named_explicitly_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, _Conn({}))

    assert mod.main(["--collection", "kb"]) == 2


def test_a_row_that_does_not_decode_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    asyncio.run(
        cm.collection_manager.save_collection(
            CollectionInfo(
                name="kb",
                embedding_model_id="BAAI/old-model",
                embedding_dimension=2,
            )
        )
    )
    conn = mod.connection()
    conn.open_table(mod.METADATA_TABLE).update(
        "name = 'kb'", {"extra_metadata": "{not json"}
    )
    # The manager reports this row as "not found"; the strict read must not.
    with pytest.raises(ValueError, match="not found"):
        cm.get_collection_sync("kb")
    with pytest.raises(mod.MigrationCliError) as raised:
        mod.collection_info("kb")
    assert not isinstance(raised.value, mod.MissingCollectionError)

    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["kb"])
    (item,) = mod.scope_report()
    assert item.error and item.skip_reason is None
    assert mod.main([]) == 1


class _FailingWhere:
    def __init__(self, table: Any) -> None:
        self._table = table

    def where(self, where: str) -> Any:
        if "'bad'" in where:
            raise OSError("metadata store unavailable")
        return self._table.search().where(where)


def test_a_metadata_store_failure_is_not_a_missing_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    for name in ("bad", "ok"):
        asyncio.run(
            cm.collection_manager.save_collection(
                CollectionInfo(
                    name=name,
                    embedding_model_id="BAAI/old-model",
                    embedding_dimension=2,
                )
            )
        )
    real = mod.connection()

    class _Broken:
        def __init__(self, table: Any) -> None:
            self._table = table

        def search(self) -> Any:
            return _FailingWhere(self._table)

        def __getattr__(self, attr: str) -> Any:
            return getattr(self._table, attr)

    class _Conn:
        def open_table(self, name: str) -> Any:
            table = real.open_table(name)
            return _Broken(table) if name == mod.METADATA_TABLE else table

        def __getattr__(self, attr: str) -> Any:
            return getattr(real, attr)

    monkeypatch.setattr(mod, "connection", lambda: _Conn())
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["bad", "ok"])

    report = {item.name: item for item in mod.scope_report()}

    assert "metadata store unavailable" in (report["bad"].error or "")
    assert report["bad"].skip_reason is None
    assert report["ok"].error is None
    assert mod.main([]) == 1


def test_a_reader_failure_does_not_stop_the_other_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["a", "b"])

    def _scope(name: str, target: Any = None, *, conn: Any, **kwargs: Any) -> Any:
        if name == "a":
            raise RuntimeError("batch read failed")
        return mod.CollectionScope(name=name)

    monkeypatch.setattr(mod, "scope_collection", _scope)
    monkeypatch.setattr(mod, "connection", lambda: object())

    report = mod.scope_report()

    assert [item.name for item in report] == ["a", "b"]
    assert "batch read failed" in (report[0].error or "")
    assert mod.main([]) == 1


def test_a_discovered_collection_that_vanished_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mod, "resolve_collection_names", lambda **kwargs: ["gone", "ok"]
    )

    def _scope(name: str, target: Any = None, *, conn: Any, **kwargs: Any) -> Any:
        if name == "gone":
            raise mod.MissingCollectionError("gone")
        return mod.CollectionScope(name=name)

    monkeypatch.setattr(mod, "scope_collection", _scope)
    monkeypatch.setattr(mod, "connection", lambda: object())

    report = mod.scope_report()

    assert report[0].skip_reason == mod.SKIP_NO_METADATA and report[0].error is None
    assert mod.main([]) == 0


def test_an_unresolvable_target_stops_the_run_before_any_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unknown(model_id: str) -> Any:
        raise mod.MigrationCliError("no such model")

    monkeypatch.setattr(mod, "resolve_target", _unknown)
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["a"])
    monkeypatch.setattr(
        mod, "scope_collection", lambda *a, **k: pytest.fail("not reached")
    )

    assert mod.main(["--model", "nope"]) == 2


def test_an_empty_scope_is_reported_as_such(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: [])

    assert mod.main([]) == 1


# --- read-only ---


def test_the_report_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row()]),
        }
    )
    _wire(monkeypatch, conn)
    before = [dict(row) for row in conn.tables[mod.METADATA_TABLE].rows]

    mod.scope_report(collections=["kb"])

    assert conn.tables[mod.METADATA_TABLE].rows == before
    assert set(conn.tables) == {mod.METADATA_TABLE, _writer()}


def test_collection_names_are_trimmed_and_deduplicated() -> None:
    assert mod.resolve_collection_names(collections=[" kb ", "kb", "other"]) == [
        "kb",
        "other",
    ]


def test_reading_is_batched(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row(f"c{index}", doc_id=f"d{index}") for index in range(2500)]
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()]), _writer(): _Table(rows)})
    _wire(monkeypatch, conn)

    assert mod.scope_collection("kb").chunks == 2500
