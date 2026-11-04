"""Upper layers must not read or delete chunk and embedding rows through the
global store; they go through ``KBCollectionHandle`` so each engine owns them.

The scan covers ``src/xagent`` outside the LanceDB store, handle and migration
code. It matches by name, so it is best effort. Code gated by
``ledger_holds_vectors()``, as an ``if`` (or ``and``) condition or as a top-level
``if not ledger_holds_vectors(): return/raise`` in a function, runs only on
LanceDB deployments and is not flagged. Existing bypasses are allowlisted one
by one; the allowlist must match the scan exactly, so a fixed bypass has to
leave it.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[6] / "src" / "xagent"
IMPLEMENTATIONS = (
    "xagent/core/tools/core/RAG_tools/storage/",
    "xagent/core/tools/core/RAG_tools/kb/collection_handle.py",
    "xagent/migrations/lancedb/",
)

# VectorIndexStore methods that read or delete chunk/embedding rows.
ROW_METHODS = frozenset(
    """
    aggregate_collection_stats aggregate_document_counts
    cascade_delete cleanup_cascade_by_scope cleanup_cascade_by_scope_async
    delete_chunk_records delete_collection_data delete_document_data
    delete_documents_data delete_embedding_records list_indexed_doc_refs
    list_table_names list_version_candidate_rows list_version_candidate_rows_async
    open_embeddings_table search_fts_async search_fts_by_model_async search_vectors
    search_vectors_async search_vectors_by_model search_vectors_by_model_async
    """.split()
)
# Generic table readers, flagged unless the table is a literal other than
# ``chunks`` or ``embeddings_*``.
TABLE_METHODS = frozenset(
    """
    count_rows count_rows_async count_rows_or_zero get_vector_dimension
    get_vector_dimension_async iter_batches iter_batches_async
    """.split()
)
RAW_ACCESSORS = frozenset(
    {"get_vector_store_raw_connection", "list_embeddings_table_names"}
)
GATE = "ledger_holds_vectors"

# Existing bypasses: how often each one occurs and what it is for.
ALLOWLIST: dict[str, dict[tuple[str, str], tuple[int, str]]] = {
    "xagent/core/tools/core/RAG_tools/utils/migration_utils.py": {
        ("migrate_embeddings_table", "get_vector_store_raw_connection"): (
            1,
            "body of LanceDBVectorIndexStore.migrate_embeddings_table",
        ),
    },
}


def _callee(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_gate(node: ast.expr) -> bool:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return any(_is_gate(value) for value in node.values)
    return isinstance(node, ast.Call) and _callee(node) == GATE


def _is_guard(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and _is_gate(node.test.operand)
        and isinstance(node.body[-1], (ast.Return, ast.Raise))
    )


def _is_other_literal_table(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value != "chunks"
        and not node.value.startswith("embeddings_")
    )


class _BypassScanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.store_names: list[set[str]] = [set()]
        self.found: Counter[tuple[str, str]] = Counter()

    def _record(self, name: str) -> None:
        self.found[(".".join(self.scope) or "<module>", name)] += 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        names = {
            arg.arg
            for arg in node.args.args + node.args.kwonlyargs
            if arg.annotation is not None
            and "VectorIndexStore" in ast.unparse(arg.annotation)
        }
        names |= {
            target.id
            for child in ast.walk(node)
            if isinstance(child, (ast.Assign, ast.AnnAssign))
            and child.value is not None
            and self._is_store(child.value)
            for target in (
                child.targets if isinstance(child, ast.Assign) else [child.target]
            )
            if isinstance(target, ast.Name)
        }
        self.scope.append(node.name)
        self.store_names.append(names | self.store_names[-1])
        for child in [*node.decorator_list, node.args]:
            self.visit(child)
        for statement in node.body:
            self.visit(statement)
            if _is_guard(statement):
                break
        self.store_names.pop()
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        if not _is_gate(node.test):
            self.generic_visit(node)
            return
        self.visit(node.test)
        for statement in node.orelse:
            self.visit(statement)

    def _is_store(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Call):
            return _callee(node) == "get_vector_index_store"
        if isinstance(node, ast.Name):
            return node.id in self.store_names[-1]
        return isinstance(node, ast.Attribute) and node.attr == "vector_index_store"

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in ROW_METHODS and self._is_store(node.value):
            self._record(node.attr)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _callee(node)
        if callee in RAW_ACCESSORS:
            self._record(callee)
        elif (
            callee in TABLE_METHODS
            and isinstance(node.func, ast.Attribute)
            and self._is_store(node.func.value)
        ):
            table = node.args[0] if node.args else None
            for keyword in node.keywords:
                if keyword.arg == "table_name":
                    table = keyword.value
            if not _is_other_literal_table(table):
                self._record(callee)
        self.generic_visit(node)


def _scan(source: str) -> Counter[tuple[str, str]]:
    scanner = _BypassScanner()
    scanner.visit(ast.parse(source))
    return scanner.found


def test_upper_layers_reach_chunks_and_embeddings_only_through_the_handle() -> None:
    found: Counter[tuple[str, str, str]] = Counter()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(SRC_ROOT.parent).as_posix()
        if not relative.startswith(IMPLEMENTATIONS):
            for hit, count in _scan(path.read_text("utf-8")).items():
                found[(relative, *hit)] += count
    allowed = Counter(
        {
            (path, *hit): count
            for path, hits in ALLOWLIST.items()
            for hit, (count, _purpose) in hits.items()
        }
    )

    assert found - allowed == Counter(), "new bypass of KBCollectionHandle"
    assert allowed - found == Counter(), "fixed bypass still allowlisted"


def test_scanner_flags_each_bypass_shape() -> None:
    source = """
def direct():
    get_vector_index_store().delete_document_data("kb", "d", 1, False)

def bound_name():
    store = get_vector_index_store()
    return store.aggregate_collection_stats

def annotated(store: VectorIndexStore):
    return store.count_rows("chunks")

def computed_table(table_name):
    store = get_vector_index_store()
    return store.count_rows_or_zero(table_name)

def context_store(context):
    return context.vector_index_store.cascade_delete(target="collection")

class Facade:
    def enumerate(self, conn):
        return list_embeddings_table_names(conn)

def closure():
    store = get_vector_index_store()

    def _compensate():
        store.delete_documents_data("kb", ["d"], None, True)

    return _compensate

def annotated_assignment():
    store: VectorIndexStore = get_vector_index_store()
    return store.list_indexed_doc_refs("kb")

def other_tables_only(facade):
    store = get_vector_index_store()
    store.count_rows("documents")
    store.iter_batches(table_name="parses")
    return facade.cascade_delete()

async def each_occurrence(store: VectorIndexStore):
    store.count_rows("chunks")
    store.count_rows("chunks")
    return await store.get_vector_dimension_async("embeddings_m")

def gated(store: VectorIndexStore, flag):
    if ledger_holds_vectors():
        store.count_rows("chunks")
    else:
        store.cascade_delete()
    if flag and ledger_holds_vectors():
        store.list_indexed_doc_refs([])

def guarded(conn):
    if not ledger_holds_vectors():
        return None
    return list_embeddings_table_names(conn)

def not_gated(store: VectorIndexStore, flag):
    if not ledger_holds_vectors():
        store.count_rows("chunks")
    if flag or ledger_holds_vectors():
        store.count_rows("chunks")
"""

    assert _scan(source) == {
        ("each_occurrence", "count_rows"): 2,
        ("each_occurrence", "get_vector_dimension_async"): 1,
        ("direct", "delete_document_data"): 1,
        ("bound_name", "aggregate_collection_stats"): 1,
        ("annotated", "count_rows"): 1,
        ("computed_table", "count_rows_or_zero"): 1,
        ("context_store", "cascade_delete"): 1,
        ("Facade.enumerate", "list_embeddings_table_names"): 1,
        ("closure._compensate", "delete_documents_data"): 1,
        ("annotated_assignment", "list_indexed_doc_refs"): 1,
        ("gated", "cascade_delete"): 1,
        ("not_gated", "count_rows"): 2,
    }
