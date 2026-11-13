"""Tests for the switch's per-row acceptance check, on a real LanceDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lancedb
import pyarrow as pa
import pytest

from xagent.migrations.embedding import acceptance as mod
from xagent.migrations.embedding import scope as scope_mod

Row = Tuple[str, str, Optional[int]]
GOOD = [0.5, 0.5]

_SCHEMA = pa.schema(
    [
        ("collection", pa.string()),
        ("doc_id", pa.string()),
        ("chunk_id", pa.string()),
        ("user_id", pa.int64()),
        ("text", pa.string()),
        ("vector", pa.list_(pa.float32())),
    ]
)


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(
        scope_mod, "source_table", lambda model_id: f"embeddings_{model_id}"
    )
    return lancedb.connect(str(tmp_path / "db"))


def _table(
    conn: Any, name: str, rows: List[Dict[str, Any]], schema: Any = None
) -> None:
    if not rows:
        conn.create_table(name, schema=schema or _SCHEMA)
        return
    full = [{"collection": "kb", "text": "t", "vector": GOOD, **row} for row in rows]
    conn.create_table(name, data=pa.Table.from_pylist(full, schema=schema or _SCHEMA))


def _rows(items: List[Row]) -> List[Dict[str, Any]]:
    return [{"doc_id": d, "chunk_id": c, "user_id": u} for d, c, u in items]


def _check(conn: Any) -> mod.AcceptanceResult:
    return mod.check_collection(
        "kb", source_model_id="old", target_model_id="new", conn=conn
    )


def test_matching_rows_pass(conn: Any) -> None:
    rows: List[Row] = [("a", "a1", 1), ("a", "a2", 1), ("b", "b1", 1)]
    _table(conn, "embeddings_old", _rows(rows))
    _table(conn, "embeddings_new", _rows(rows))

    result = _check(conn)

    assert result.ok is True
    assert result.checks["source_rows"] == 3
    assert result.checks["target_rows"] == 3


def test_other_collections_are_not_compared(conn: Any) -> None:
    _table(conn, "embeddings_old", _rows([("a", "a1", 1)]))
    _table(
        conn,
        "embeddings_new",
        _rows([("a", "a1", 1)])
        + [{"collection": "other", **_rows([("z", "z1", 1)])[0]}],
    )

    assert _check(conn).ok is True


def test_an_empty_collection_switches_without_a_target_table(conn: Any) -> None:
    _table(conn, "embeddings_old", [])

    assert _check(conn).ok is True


def test_nothing_copied_for_a_non_empty_source_fails(conn: Any) -> None:
    _table(conn, "embeddings_old", _rows([("a", "a1", 1)]))

    result = _check(conn)

    assert result.ok is False
    assert result.checks["row_mismatch_count"] == 1


def test_a_missing_source_table_fails(conn: Any) -> None:
    _table(conn, "embeddings_new", _rows([("a", "a1", 1)]))

    result = _check(conn)

    assert result.ok is False
    assert "does not exist" in result.failures[0]


def test_both_sides_read_the_writer_table(conn: Any) -> None:
    _table(conn, "embeddings_old", _rows([("a", "a1", 1), ("a", "a2", 1)]))
    _table(conn, "embeddings_new", _rows([("a", "a1", 1)]))

    result = _check(conn)

    assert result.ok is False
    assert result.checks["source_rows"] == 2


def test_ownerless_rows_are_left_out_of_both_sides(conn: Any) -> None:
    _table(conn, "embeddings_old", _rows([("a", "a1", 1), ("a", "l1", None)]))
    # Rollback direction: the old table holds a legacy ownerless row too.
    _table(conn, "embeddings_new", _rows([("a", "a1", 1), ("b", "l2", None)]))

    result = _check(conn)

    assert result.ok is True
    assert result.checks["excluded_no-owner"] == 1
    assert result.checks["target_excluded_no-owner"] == 1


def test_a_textless_row_excludes_only_itself(conn: Any) -> None:
    source = _rows([("a", "a1", 1), ("a", "a2", 1), ("a", "a3", 1)])
    source[2]["text"] = ""
    _table(conn, "embeddings_old", source)
    # a2 is missing from the target: the textless a3 must not hide that.
    _table(conn, "embeddings_new", _rows([("a", "a1", 1)]))

    result = _check(conn)

    assert result.ok is False
    assert result.checks["row_mismatches"] == ["a:a2"]
    assert result.checks["excluded_no-text"] == 1


def test_a_null_text_row_is_excluded(conn: Any) -> None:
    source = _rows([("a", "a1", 1), ("a", "a2", 1)])
    source[1]["text"] = None
    _table(conn, "embeddings_old", source)
    _table(conn, "embeddings_new", _rows([("a", "a1", 1)]))

    result = _check(conn)

    assert result.ok is True
    assert result.checks["excluded_no-text"] == 1


def test_a_row_missing_from_the_target_fails(conn: Any) -> None:
    _table(conn, "embeddings_old", _rows([("a", "a1", 1), ("a", "a2", 1)]))
    _table(conn, "embeddings_new", _rows([("a", "a1", 1)]))

    result = _check(conn)

    assert result.ok is False
    assert result.checks["row_mismatches"] == ["a:a2"]


def test_an_extra_target_row_fails(conn: Any) -> None:
    _table(conn, "embeddings_old", _rows([("a", "a1", 1)]))
    _table(conn, "embeddings_new", _rows([("a", "a1", 1), ("a", "a2", 1)]))

    assert _check(conn).ok is False


def test_a_row_with_another_owner_fails(conn: Any) -> None:
    _table(conn, "embeddings_old", _rows([("a", "a1", 1)]))
    _table(conn, "embeddings_new", _rows([("a", "a1", 2)]))

    result = _check(conn)

    assert result.ok is False
    assert result.checks["row_mismatches"] == ["a:a1"]


# LanceDB itself rejects NaN and ragged vectors on write; these still land.
@pytest.mark.parametrize("vector", [None, [0.0, 0.0], [float("inf"), 1.0]])
def test_an_unusable_target_vector_fails(conn: Any, vector: Any) -> None:
    _table(conn, "embeddings_old", _rows([("a", "a1", 1), ("a", "a2", 1)]))
    target = _rows([("a", "a1", 1), ("a", "a2", 1)])
    target[1]["vector"] = vector
    _table(conn, "embeddings_new", target)

    result = _check(conn)

    assert result.ok is False
    assert result.checks["target_bad_vectors"] == 1


def test_a_bad_vector_on_an_excluded_row_is_ignored(conn: Any) -> None:
    _table(conn, "embeddings_old", _rows([("a", "a1", 1)]))
    target = _rows([("a", "a1", 1), ("legacy", "l1", None)])
    target[1]["vector"] = None
    _table(conn, "embeddings_new", target)

    result = _check(conn)

    assert result.ok is True
    assert result.checks["target_bad_vectors"] == 0


def test_a_failed_read_raises_instead_of_passing(
    conn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _table(conn, "embeddings_old", _rows([("a", "a1", 1)]))

    def _broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("corrupt table")

    monkeypatch.setattr(mod, "iter_rows", _broken)

    with pytest.raises(RuntimeError, match="corrupt table"):
        _check(conn)


def test_a_table_without_a_text_column_is_a_hard_failure(conn: Any) -> None:
    schema = pa.schema([f for f in _SCHEMA if f.name != "text"])
    conn.create_table(
        "embeddings_old",
        data=pa.Table.from_pylist(
            [
                {
                    "collection": "kb",
                    "doc_id": "a",
                    "chunk_id": "a1",
                    "user_id": 1,
                    "vector": GOOD,
                }
            ],
            schema=schema,
        ),
    )

    with pytest.raises(mod.MigrationCliError, match="no text column"):
        _check(conn)
