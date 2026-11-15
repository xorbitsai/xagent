"""Scope: eligibility, strict reads and per-collection isolation."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
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
        ("model", pa.string()),
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
    model: Optional[str] = None,
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
        "model": model,
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
        _require_fields(self.table, self.filter)
        rows = [row for row in self.table.rows if _matches(row, self.filter)]
        for start in range(0, len(rows), batch_size):
            yield pa.RecordBatch.from_pylist(
                [
                    {key: row.get(key) for key in self.columns or row}
                    for row in rows[start : start + batch_size]
                ]
            )

    def to_arrow(self) -> Any:
        _require_fields(self.table, self.filter)
        return pa.Table.from_pylist(
            [row for row in self.table.rows if _matches(row, self.filter)]
        )


def _require_fields(table: "_Table", where: str) -> None:
    # LanceDB rejects a filter on a column the table lacks; so must the double.
    for name in re.findall(r"(\w+) (?:=|!=|IS NULL)", where or ""):
        if name not in table.schema.names:
            raise RuntimeError(f"No field named {name}")


def _matches(row: Dict[str, Any], where: str) -> bool:
    if not where or where == "true":
        return True
    return all(_matches_clause(row, part) for part in where.split(" AND "))


def _matches_clause(row: Dict[str, Any], clause: str) -> bool:
    if clause.startswith("(") and clause.endswith(")"):
        return any(_matches_clause(row, part) for part in clause[1:-1].split(" OR "))
    if clause.endswith(" IS NULL"):
        return row.get(clause[: -len(" IS NULL")].strip()) is None
    for operator in (" != ", " = "):
        if operator in clause:
            column, _, literal = clause.partition(operator)
            value = row.get(column.strip())
            if operator == " != ":
                # SQL: NULL is not "different from", it is unknown.
                return value is not None and str(value) != literal.strip("'")
            return str(value) == literal.strip("'")
    raise AssertionError(f"unsupported filter clause: {clause!r}")


class _Table:
    def __init__(self, rows: List[Dict[str, Any]], names: Optional[List[str]] = None):
        self.rows = rows
        self.schema = pa.schema(
            [(name, pa.string()) for name in (names or list(SCHEMA.names))]
        )

    def search(self) -> _Query:
        return _Query(self)

    def count_rows(self, filter: Optional[str] = None) -> int:
        _require_fields(self, filter or "")
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
    assert (item.usable_documents, item.chunks) == (1, 2)
    assert mod.describe(item).startswith("kb: source model BAAI/old-model")
    assert "missing" not in mod.describe(item)


def test_an_eligible_collection_without_rows_is_counted_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()])})
    _wire(monkeypatch, conn)

    item = mod.scope_collection("kb")

    assert item.usable and (item.usable_documents, item.chunks) == (0, 0)
    assert mod.precompute_eligibility("kb").model_id == "BAAI/old-model"


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

    assert mod.precompute_eligibility("kb").reason == mod.SKIP_UNBOUND
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

    assert mod.precompute_eligibility("kb").reason == mod.SKIP_BOUND_WITHOUT_DIMENSION


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
    assert mod.precompute_eligibility("kb").reason == mod.SKIP_ROWS_IN_OTHER_TABLE


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

    assert mod.precompute_eligibility("kb").reason == mod.SKIP_ROWS_IN_MULTIPLE_TABLES


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

    assert mod.precompute_eligibility("kb").reason is None


@pytest.mark.parametrize(
    "column", ["collection", "doc_id", "chunk_id", "user_id", "text"]
)
def test_a_missing_required_column_is_skipped(
    monkeypatch: pytest.MonkeyPatch, column: str
) -> None:
    table = _Table([_row()], names=[name for name in SCHEMA.names if name != column])
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()]), _writer(): table})
    _wire(monkeypatch, conn)

    assert mod.precompute_eligibility("kb").reason == mod.SKIP_MISSING_COLUMN


def test_another_engine_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()])})
    _wire(monkeypatch, conn, engine="milvus")

    assert mod.precompute_eligibility("kb").reason == mod.SKIP_OTHER_ENGINE


def test_an_unreadable_engine_binding_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()])})
    _wire(monkeypatch, conn, engine="")

    assert mod.precompute_eligibility("kb").reason == mod.SKIP_UNREADABLE_BINDING


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
        mod.SKIP_NO_DOC_ID: 1,
        mod.SKIP_NO_CHUNK_ID: 1,
        mod.SKIP_NO_OWNER: 1,
        mod.SKIP_NO_TEXT: 1,
    }
    assert item.usable_documents == 2
    assert "2 usable documents" in mod.describe(item)


def test_the_missing_count_uses_the_target_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row("c1"), _row("c2"), _row("c3")]),
            mod.source_table("new-model"): _Table([_row("c1", model="new-model")]),
        }
    )
    _wire(monkeypatch, conn)

    assert mod.scope_collection("kb", target="new-model").missing_chunks == 2


@pytest.mark.parametrize("vector", [[0.0, 0.0], [float("nan"), 0.5]])
def test_a_target_row_with_a_bad_vector_is_still_missing(
    monkeypatch: pytest.MonkeyPatch, vector: List[float]
) -> None:
    bad = _row("c1", model="new-model")
    bad["vector"] = vector
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row("c1")]),
            mod.source_table("new-model"): _Table([bad]),
        }
    )
    _wire(monkeypatch, conn)

    assert mod.scope_collection("kb", target="new-model").missing_chunks == 1


def test_a_chunk_id_held_under_another_document_is_still_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row("c1", doc_id="d1"), _row("c1", doc_id="d2")]),
            mod.source_table("new-model"): _Table(
                [_row("c1", doc_id="d1", model="new-model")]
            ),
        }
    )
    _wire(monkeypatch, conn)

    assert mod.scope_collection("kb", target="new-model").missing_chunks == 1


def test_only_rows_the_target_wrote_in_its_writer_table_are_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.source_table("new-model"): _Table(
                [
                    _row("c1", model="new-model"),
                    _row("c2", model="new_model"),
                    _row("c4", model="new-model "),
                ]
            ),
            "embeddings_new_legacy": _Table([_row("c3", model="new-model")]),
        }
    )
    _wire(monkeypatch, conn)

    held = mod.target_chunk_ids("kb", "new-model", dimension=2)

    assert held == {("d1", "c1")}


@pytest.mark.parametrize(
    ("target_rows", "missing"), [([], 1), ([_row("c1", model="new-model")], 0)]
)
def test_a_coreless_row_is_excluded_not_missing(
    monkeypatch: pytest.MonkeyPatch, target_rows: List[Dict[str, Any]], missing: int
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row("c1"), _row("c2")]),
            mod.CORE_TABLE: _Table(
                [{"collection": "kb", "doc_id": "d1", "chunk_id": "c1"}],
                names=["collection", "doc_id", "chunk_id"],
            ),
            mod.source_table("new-model"): _Table(target_rows),
        }
    )
    _wire(monkeypatch, conn)

    item = mod.scope_collection("kb", target="new-model")

    assert item.chunks == 2
    assert item.missing_chunks == missing
    assert f"{missing} missing for the target model" in mod.describe(item)
    assert item.skipped == {mod.SKIP_NO_CORE_CHUNK: 1}


def _core(*keys: Any) -> _Table:
    return _Table(
        [{"collection": "kb", "doc_id": d, "chunk_id": c} for d, c in keys],
        names=["collection", "doc_id", "chunk_id"],
    )


def test_rows_of_a_document_the_core_table_lacks_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row("c1", doc_id="d1"), _row("c2", doc_id="d2")]),
            mod.CORE_TABLE: _core(("d1", "c1")),
        }
    )
    _wire(monkeypatch, conn)

    item = mod.scope_collection("kb", target="new-model")

    assert (item.missing_chunks, item.skipped) == (2, {})


def test_ids_are_compared_as_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _Table([_row("c1", doc_id="d1 ")])
    held = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): source,
            mod.source_table("new-model"): _Table(
                [_row("c1", doc_id="d1", model="new-model")]
            ),
        }
    )
    _wire(monkeypatch, held)
    assert mod.scope_collection("kb", target="new-model").missing_chunks == 1

    backed = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): source,
            mod.CORE_TABLE: _core(("d1 ", "c2")),
        }
    )
    _wire(monkeypatch, backed)
    assert mod.scope_collection("kb").skipped == {mod.SKIP_NO_CORE_CHUNK: 1}


# --- strict reads and isolation ---


def test_an_absent_table_or_row_is_a_missing_collection() -> None:
    with pytest.raises(mod.MissingCollectionError):
        mod.collection_info("kb")
    _save_collection()
    with pytest.raises(mod.MissingCollectionError):
        mod.collection_info("other")


def test_a_missing_collection_named_explicitly_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, _Conn({}))

    assert mod.main(["--collection", "kb"]) == 2


def test_an_unknown_name_stops_the_run_before_any_collection_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, _Conn({mod.METADATA_TABLE: _metadata([_bound()])}))
    scoped: List[str] = []
    monkeypatch.setattr(
        mod, "scope_collection", lambda name, *a, **k: scoped.append(name)
    )

    assert mod.main(["--collection", "kb", "--collection", "nope"]) == 2
    assert scoped == []


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


@pytest.mark.parametrize("setting", ["qdrant", "no-such-engine"])
def test_an_unusable_backend_setting_is_an_error_not_a_skip(
    monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()])})
    monkeypatch.setattr(mod, "connection", lambda: conn)
    monkeypatch.setenv("XAGENT_VECTOR_BACKEND", setting)

    (item,) = mod.scope_report(collections=["kb"])

    assert item.error and item.skip_reason is None


def test_a_malformed_engine_binding_is_unreadable() -> None:
    from types import SimpleNamespace

    from xagent.core.tools.core.RAG_tools.kb.coordinator import (
        KB_STORAGE_METADATA_KEY,
    )

    info = SimpleNamespace(extra_metadata={KB_STORAGE_METADATA_KEY: "bogus"})

    assert mod.collection_engine(info) == ""


def test_a_coordinator_that_fails_to_build_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from xagent.core.tools.core.RAG_tools.kb import coordinator

    def _broken() -> Any:
        raise ValueError("cannot build the coordinator")

    monkeypatch.setattr(coordinator, "get_kb_coordinator", _broken)

    with pytest.raises(ValueError):
        mod.collection_engine(SimpleNamespace(extra_metadata={}))


def test_a_reader_failure_does_not_stop_the_other_collections(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
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
    assert caplog.text.count("batch read failed") == 1


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


def test_an_empty_scope_is_reported_as_such(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: [])

    assert mod.main([]) == 1


# --- the store directory and the arguments ---


@pytest.mark.parametrize("argv", [["--collection", "kb"], []])
@pytest.mark.parametrize("configured", [True, False])
def test_a_missing_lancedb_dir_is_left_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: List[str],
    configured: bool,
) -> None:
    from xagent.providers.vector_store.lancedb import LanceDBConnectionManager

    missing = tmp_path / "absent" / "lancedb"
    if configured:
        monkeypatch.setenv("LANCEDB_DIR", str(missing))
    else:
        monkeypatch.delenv("LANCEDB_DIR", raising=False)
        monkeypatch.setattr(
            LanceDBConnectionManager,
            "locate_default_lancedb_dir",
            staticmethod(lambda: missing),
        )
    models = tmp_path / "language_models"
    models.mkdir()
    monkeypatch.setenv("LANCE_LANGUAGE_MODEL_HOME", str(models))

    assert mod.main(argv) == 2

    assert not (tmp_path / "absent").exists()
    assert list(models.iterdir()) == []


def test_the_application_still_creates_its_default_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.providers.vector_store.lancedb import LanceDBConnectionManager

    default = tmp_path / "default" / "lancedb"
    monkeypatch.delenv("LANCEDB_DIR", raising=False)
    monkeypatch.setattr(
        LanceDBConnectionManager,
        "locate_default_lancedb_dir",
        staticmethod(lambda: default),
    )
    cached = LanceDBConnectionManager.get_default_lancedb_dir
    cached.cache_clear()
    try:
        assert LanceDBConnectionManager().resolve_dir_from_env(create=False)
        assert not default.exists()
        assert LanceDBConnectionManager().resolve_dir_from_env() == str(default)
        assert default.is_dir()
    finally:
        cached.cache_clear()


def _unreadable(*args: Any, **kwargs: Any) -> Any:
    raise OSError("unreadable")


@pytest.mark.parametrize(
    ("patched", "argv"),
    [
        ("connection", []),
        ("_all_collection_names", []),
        (None, ["--collection", "  "]),
    ],
)
def test_a_run_that_cannot_start_exits_2(
    monkeypatch: pytest.MonkeyPatch, patched: Optional[str], argv: List[str]
) -> None:
    _wire(monkeypatch, _Conn({mod.METADATA_TABLE: _metadata([_bound()])}))
    if patched:
        monkeypatch.setattr(mod, patched, _unreadable)

    assert mod.main(argv) == 2


@pytest.mark.parametrize("model", ["", "  ", "none"])
def test_a_blank_target_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    _wire(monkeypatch, _Conn({mod.METADATA_TABLE: _metadata([_bound()])}))

    assert mod.main(["--model", model]) == 2


def test_a_target_id_is_normalised_like_a_binding() -> None:
    assert mod.resolve_target(" new-model ") == "new-model"


def test_collection_names_are_matched_as_stored() -> None:
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound(), _bound(name=" kb")])})

    assert mod.resolve_collection_names(collections=["kb", " kb", "kb"], conn=conn) == [
        "kb",
        " kb",
    ]
    with pytest.raises(mod.MissingCollectionError):
        mod.resolve_collection_names(collections=["kb "], conn=conn)


def test_reading_is_batched(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row(f"c{index}", doc_id=f"d{index}") for index in range(2500)]
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()]), _writer(): _Table(rows)})
    _wire(monkeypatch, conn)
    sizes: List[int] = []
    batches = _Query.to_batches

    def _spy(query: _Query, batch_size: int) -> Iterator[Any]:
        sizes.append(batch_size)
        return batches(query, batch_size)

    monkeypatch.setattr(_Query, "to_batches", _spy)

    assert mod.scope_collection("kb").chunks == 2500
    assert mod.READ_BATCH_SIZE in sizes


# --- imports ---


def test_every_module_in_the_migration_package_imports() -> None:
    import importlib
    import pkgutil

    import xagent.migrations.embedding as package

    names = {module.name for module in pkgutil.iter_modules(package.__path__)}
    assert "scope" in names
    for name in sorted(names):
        importlib.import_module(f"xagent.migrations.embedding.{name}")


# --- source and target must not share a physical table ---


def test_a_target_that_writes_the_source_table_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        written_embeddings_table_name,
    )

    # ``old-model`` and ``old_model`` share one writer spelling.
    assert written_embeddings_table_name("old-model") == written_embeddings_table_name(
        "old_model"
    )
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound("old-model", 2)]),
            mod.source_table("old-model"): _Table([_row("c1")]),
        }
    )
    _wire(monkeypatch, conn)

    assert (
        mod.precompute_eligibility("kb", target_model_id="old_model").reason
        == mod.SKIP_SHARED_TABLE
    )
    item = mod.scope_collection("kb", target="old_model")
    assert item.skip_reason == mod.SKIP_SHARED_TABLE
    assert item.chunks == 0


def test_a_target_that_is_the_bound_model_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row("c1")]),
        }
    )
    _wire(monkeypatch, conn)

    item = mod.scope_collection("kb", target="BAAI/old-model")

    assert item.skip_reason == mod.SKIP_TARGET_IS_SOURCE
    assert item.missing_chunks is None


def test_the_refusal_happens_before_anything_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound("old-model", 2)]),
            mod.source_table("old-model"): _Table([_row("c1")]),
        }
    )
    _wire(monkeypatch, conn)
    monkeypatch.setattr(mod, "resolve_target", lambda model_id: "old_model")
    monkeypatch.setattr(mod, "resolve_collection_names", lambda **kwargs: ["kb"])

    report = mod.scope_report(target_model_id="old_model")

    assert [item.error for item in report] == [None]
    assert report[0].skip_reason == mod.SKIP_SHARED_TABLE
    assert report[0].chunks == 0


# --- a target table whose rows cannot be attributed to a collection ---

_NO_COLLECTION_COLUMN = ["doc_id", "chunk_id", "user_id", "text", "model", "vector"]


def test_a_target_table_that_cannot_name_its_collection_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without the column a row cannot be proven foreign.
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row("c1")]),
            mod.source_table("new-model"): _Table(
                [_row("c9")], names=_NO_COLLECTION_COLUMN
            ),
        }
    )
    _wire(monkeypatch, conn)

    assert (
        mod.precompute_eligibility("kb", target_model_id="new-model").reason
        == mod.SKIP_TARGET_ROWS_IN_OTHER_TABLE
    )


def test_a_target_row_no_model_wrote_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row("c1")]),
            mod.source_table("new-model"): _Table([_row("c9", model=None)]),
        }
    )
    _wire(monkeypatch, conn)

    assert (
        mod.precompute_eligibility("kb", target_model_id="new-model").reason
        == mod.SKIP_TARGET_ROWS_IN_OTHER_TABLE
    )


def test_an_empty_target_table_without_the_column_does_not_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table([_row("c1")]),
            mod.source_table("new-model"): _Table([], names=_NO_COLLECTION_COLUMN),
        }
    )
    _wire(monkeypatch, conn)

    assert mod.precompute_eligibility("kb", target_model_id="new-model").reason is None
    assert mod.scope_collection("kb", target="new-model").missing_chunks == 1


def test_an_empty_collection_with_an_absent_source_table_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()])})
    _wire(monkeypatch, conn)

    eligibility = mod.precompute_eligibility("kb", target_model_id="new-model")

    assert eligibility.reason is None
    assert mod.scope_collection("kb", target="new-model").missing_chunks == 0


# --- one pass over the source rows ---


def test_the_scan_classifies_every_row_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn(
        {
            mod.METADATA_TABLE: _metadata([_bound()]),
            _writer(): _Table(
                [
                    _row("c1", doc_id="d1"),
                    _row("c2", doc_id="d2"),
                    _row("c3", doc_id="d1"),
                    _row("c4", doc_id=None),
                    _row("c5", doc_id="d3", user_id=None),
                ]
            ),
        }
    )
    _wire(monkeypatch, conn)

    scan = mod.scan_source("kb", "BAAI/old-model", have={("d2", "c2")})

    assert scan.usable_documents == {"d1", "d2"}
    assert scan.pending == 2
    assert scan.skipped == {mod.SKIP_NO_DOC_ID: 1, mod.SKIP_NO_OWNER: 1}


def test_a_missing_source_table_scans_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn({mod.METADATA_TABLE: _metadata([_bound()])})
    _wire(monkeypatch, conn)

    scan = mod.scan_source("kb", "BAAI/old-model", have=set())

    assert (scan.rows, scan.pending) == (0, 0)


# --- the same readers on a real LanceDB store ---

_CORE_SCHEMA = pa.schema(
    [
        ("collection", pa.string()),
        ("doc_id", pa.string()),
        ("chunk_id", pa.string()),
        ("user_id", pa.int64()),
    ]
)


def _save_collection() -> None:
    import asyncio

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    asyncio.run(
        cm.collection_manager.save_collection(
            CollectionInfo(
                name="kb",
                embedding_model_id="BAAI/old-model",
                embedding_dimension=2,
                embeddings=4,
            )
        )
    )


def _real_table(name: str, rows: List[Dict[str, Any]], schema: Any = SCHEMA) -> None:
    mod.connection().create_table(name, data=pa.Table.from_pylist(rows, schema=schema))


def _versions() -> Dict[str, int]:
    conn = mod.connection()
    return {name: conn.open_table(name).version for name in mod.table_names(conn)}


@pytest.fixture
def real_store() -> None:
    _save_collection()


def test_a_real_store_is_counted_and_left_as_it_was(real_store: None) -> None:
    _real_table(
        mod.source_table("BAAI/old-model"),
        [
            _row("c1", model="BAAI/old-model"),
            _row("c2", model="BAAI/old-model"),
            _row("c3", model="BAAI/old-model", user_id=None),
        ],
    )
    _real_table(
        mod.CORE_TABLE,
        [
            {"collection": "kb", "doc_id": "d1", "chunk_id": chunk_id, "user_id": 7}
            for chunk_id in ("c1", "c2", "c3")
        ],
        schema=_CORE_SCHEMA,
    )
    _real_table(mod.source_table("new-model"), [_row("c1", model="new-model")])
    before = _versions()

    item = mod.scope_collection("kb", target="new-model")

    assert (item.skip_reason, item.error) == (None, None)
    assert (item.chunks, item.missing_chunks) == (3, 1)
    assert item.skipped == {mod.SKIP_NO_OWNER: 1}
    assert _versions() == before


def test_a_row_another_model_wrote_in_the_target_table_is_refused_on_a_real_store(
    real_store: None,
) -> None:
    # ``new-model`` and ``new_model`` are distinct bindings with one writer table.
    _real_table(
        mod.source_table("BAAI/old-model"), [_row("c1", model="BAAI/old-model")]
    )
    _real_table(mod.source_table("new-model"), [_row("c9", model="new_model")])

    eligibility = mod.precompute_eligibility("kb", target_model_id="new-model")

    assert eligibility.reason == mod.SKIP_TARGET_ROWS_IN_OTHER_TABLE


def test_a_vector_dimension_is_checked_only_when_given_on_a_real_store(
    real_store: None,
) -> None:
    wide = pa.schema(
        [
            f if f.name != "vector" else pa.field("vector", pa.list_(pa.float32(), 3))
            for f in SCHEMA
        ]
    )
    _real_table(
        mod.source_table("BAAI/old-model"), [_row("c1", model="BAAI/old-model")]
    )
    target = _row("c1", model="new-model")
    target["vector"] = [0.5, 0.5, 0.5]
    _real_table(mod.source_table("new-model"), [target], schema=wide)

    assert mod.target_chunk_ids("kb", "new-model", dimension=2) == set()
    assert mod.target_chunk_ids("kb", "new-model") == {("d1", "c1")}
    assert mod.scope_collection("kb", target="new-model").missing_chunks == 0


@pytest.mark.parametrize("holds_rows", [True, False])
def test_a_source_table_without_a_collection_column_is_refused_on_a_real_store(
    real_store: None, holds_rows: bool
) -> None:
    narrow = pa.schema([f for f in SCHEMA if f.name != "collection"])
    row = _row("c1", model="BAAI/old-model")
    del row["collection"]
    rows = [row] if holds_rows else []
    _real_table(mod.source_table("BAAI/old-model"), rows, schema=narrow)

    item = mod.scope_collection("kb")

    assert (item.skip_reason, item.error) == (mod.SKIP_MISSING_COLUMN, None)


@pytest.mark.parametrize("missing", ["doc_id", "chunk_id"])
@pytest.mark.parametrize("table", ["core", "target"])
def test_a_table_without_a_compared_column_is_an_error_on_a_real_store(
    real_store: None, missing: str, table: str
) -> None:
    _real_table(
        mod.source_table("BAAI/old-model"), [_row("c1", model="BAAI/old-model")]
    )
    if table == "core":
        schema = pa.schema([f for f in _CORE_SCHEMA if f.name != missing])
        row: Dict[str, Any] = {"collection": "kb", "doc_id": "d1", "chunk_id": "c1"}
        name = mod.CORE_TABLE
    else:
        schema = pa.schema([f for f in SCHEMA if f.name != missing])
        row = _row("c1", model="new-model")
        name = mod.source_table("new-model")
    row.pop(missing)
    _real_table(name, [row], schema=schema)

    (item,) = mod.scope_report(collections=["kb"], target_model_id="new-model")

    assert item.error and missing in item.error and item.skip_reason is None


def test_more_rows_than_one_page_are_all_read_on_a_real_store(
    real_store: None,
) -> None:
    keys = [f"c{index:02d}" for index in range(25)]
    _real_table(
        mod.source_table("BAAI/old-model"),
        [_row(key, model="BAAI/old-model") for key in keys],
    )
    _real_table(
        mod.CORE_TABLE,
        [
            {"collection": "kb", "doc_id": "d1", "chunk_id": key, "user_id": 7}
            for key in keys
        ],
        schema=_CORE_SCHEMA,
    )
    _real_table(
        mod.source_table("new-model"),
        [_row(key, model="new-model") for key in keys[:20]],
    )

    item = mod.scope_collection("kb", target="new-model")

    assert (item.chunks, item.missing_chunks, item.skipped) == (25, 5, {})


# --- the table rule: the model's own tables, else every embeddings table ---

STRAY = "embeddings_legacy_name"


def test_rows_only_in_a_table_the_model_does_not_own_are_refused_on_a_real_store(
    real_store: None,
) -> None:
    _real_table(STRAY, [_row("c1", model="legacy")])

    assert mod.scope_collection("kb").skip_reason == mod.SKIP_ROWS_IN_OTHER_TABLE


def test_without_an_own_table_other_collections_do_not_refuse_on_a_real_store(
    real_store: None,
) -> None:
    _real_table(STRAY, [_row("c1", collection="other", model="legacy")])

    item = mod.scope_collection("kb")

    assert (item.skip_reason, item.error, item.chunks) == (None, None, 0)


def test_an_own_table_hides_every_other_table_on_a_real_store(
    real_store: None,
) -> None:
    _real_table(
        mod.source_table("BAAI/old-model"), [_row("c1", model="BAAI/old-model")]
    )
    _real_table(STRAY, [_row("c9", model="legacy")])

    item = mod.scope_collection("kb")

    assert (item.skip_reason, item.chunks) == (None, 1)


def test_a_new_target_is_refused_when_another_table_holds_the_collection(
    real_store: None,
) -> None:
    _real_table(
        mod.source_table("BAAI/old-model"), [_row("c1", model="BAAI/old-model")]
    )
    _real_table(STRAY, [_row("c9", model="legacy")])

    item = mod.scope_collection("kb", target="new-model")

    assert item.skip_reason == mod.SKIP_TARGET_ROWS_IN_OTHER_TABLE


def test_a_target_table_that_is_not_the_writer_is_refused_on_a_real_store(
    real_store: None,
) -> None:
    once, writer = mod.own_tables("ACME/new-model")[::-1]
    assert once != writer
    _real_table(
        mod.source_table("BAAI/old-model"), [_row("c1", model="BAAI/old-model")]
    )
    _real_table(once, [_row("c1", model="ACME/new-model")])

    item = mod.scope_collection("kb", target="ACME/new-model")

    assert item.skip_reason == mod.SKIP_TARGET_ROWS_IN_OTHER_TABLE


def test_a_table_without_a_collection_column_holds_the_collection_on_a_real_store(
    real_store: None,
) -> None:
    row = _row("c1", model="legacy")
    del row["collection"]
    _real_table(
        STRAY, [row], schema=pa.schema([f for f in SCHEMA if f.name != "collection"])
    )

    assert mod.scope_collection("kb").skip_reason == mod.SKIP_ROWS_IN_OTHER_TABLE


def test_a_target_writer_table_without_a_model_column_is_refused_on_a_real_store(
    real_store: None,
) -> None:
    _real_table(
        mod.source_table("BAAI/old-model"), [_row("c1", model="BAAI/old-model")]
    )
    row = _row("c1")
    del row["model"]
    narrow = pa.schema([f for f in SCHEMA if f.name != "model"])
    _real_table(mod.source_table("new-model"), [row], schema=narrow)

    item = mod.scope_collection("kb", target="new-model")

    assert (item.skip_reason, item.error) == (mod.SKIP_TARGET_ROWS_IN_OTHER_TABLE, None)


def test_another_own_target_table_is_refused_beside_the_writer_on_a_real_store(
    real_store: None,
) -> None:
    writer, once = mod.own_tables("ACME/new-model")
    _real_table(
        mod.source_table("BAAI/old-model"), [_row("c1", model="BAAI/old-model")]
    )
    _real_table(writer, [_row("c1", model="ACME/new-model")])
    _real_table(once, [_row("c9", model="ACME/new-model")])

    item = mod.scope_collection("kb", target="ACME/new-model")

    assert item.skip_reason == mod.SKIP_TARGET_ROWS_IN_OTHER_TABLE


def test_a_first_run_counts_every_row_as_missing_on_a_real_store(
    real_store: None,
) -> None:
    _real_table(
        mod.source_table("BAAI/old-model"),
        [_row("c1", model="BAAI/old-model"), _row("c2", model="BAAI/old-model")],
    )
    _real_table(STRAY, [_row("c9", collection="other", model="legacy")])

    item = mod.scope_collection("kb", target="new-model")

    assert (item.skip_reason, item.missing_chunks) == (None, 2)


def test_an_existing_target_table_hides_every_other_table_on_a_real_store(
    real_store: None,
) -> None:
    _real_table(
        mod.source_table("BAAI/old-model"),
        [_row("c1", model="BAAI/old-model"), _row("c2", model="BAAI/old-model")],
    )
    _real_table(mod.source_table("new-model"), [_row("c1", model="new-model")])
    _real_table(STRAY, [_row("c9", model="legacy")])

    item = mod.scope_collection("kb", target="new-model")

    assert (item.skip_reason, item.missing_chunks) == (None, 1)


def test_a_previous_model_can_be_the_target_on_a_real_store(
    real_store: None,
) -> None:
    _real_table(
        mod.source_table("BAAI/old-model"),
        [_row("c1", model="BAAI/old-model"), _row("c2", model="BAAI/old-model")],
    )
    _real_table(mod.source_table("prev-model"), [_row("c1", model="prev-model")])

    item = mod.scope_collection("kb", target="prev-model")

    assert (item.skip_reason, item.missing_chunks) == (None, 1)


# --- the command never opens the SQL database ---


def test_the_command_runs_without_the_sql_database(
    real_store: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlalchemy

    _real_table(
        mod.source_table("BAAI/old-model"), [_row("c1", model="BAAI/old-model")]
    )
    _real_table(mod.source_table("new-model"), [_row("c1", model="new-model")])
    database = tmp_path / "absent.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database}")

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the SQL database was opened")

    monkeypatch.setattr(sqlalchemy, "create_engine", _refuse)
    monkeypatch.setattr(sqlite3, "connect", _refuse)

    assert mod.main(["--model", "new-model"]) == 0
    assert mod.main([]) == 0
    assert not database.exists()
