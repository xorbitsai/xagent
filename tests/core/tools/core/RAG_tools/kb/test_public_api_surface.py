"""Issue #507 - Public API surface guard tests.

Every retained KB/RAG public symbol is asserted importable from its declared
path. Every retained function is asserted to keep its current sync/async shape.

This test is the executable form of the C13 public-surface audit. If a symbol
is removed, renamed, or changes sync/async shape, this test will fail before
Phase 2 handle-replacement issues (#508-#514) can merge.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

import xagent.core.tools.core.RAG_tools as rag_tools_pkg

# Each entry: (module_dotted, symbol, kind)
# kind: "sync"=sync function, "async"=async function, "class"=class, "value"=non-callable
PUBLIC_SURFACE: list[tuple[str, str, str]] = [
    ("xagent.core.tools.core.RAG_tools.chunk.__init__", "chunk_document", "sync"),
    ("xagent.core.tools.core.RAG_tools.chunk.__init__", "chunk_fixed_size", "sync"),
    ("xagent.core.tools.core.RAG_tools.chunk.__init__", "chunk_markdown", "sync"),
    ("xagent.core.tools.core.RAG_tools.chunk.__init__", "chunk_recursive", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.generate.__init__",
        "format_generation_prompt",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager",
        "check_table_needs_migration",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager",
        "ensure_chunks_table",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager",
        "ensure_collection_config_table",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager",
        "ensure_collection_metadata_table",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager",
        "ensure_documents_table",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager",
        "ensure_embeddings_table",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager",
        "ensure_ingestion_runs_table",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager",
        "ensure_main_pointers_table",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager",
        "ensure_parses_table",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager",
        "ensure_prompt_templates_table",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "CollectionConfigSnapshot",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "CollectionRollbackMaintenanceResult",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "CompensationStep", "class"),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "KBAccessMode", "class"),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBApiCompatibilityFacade",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBApiFailedIngestCleanupDecision",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBApiFailedIngestRollbackResult",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "KBApiOperationResult", "class"),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "KBBackendCapabilities", "class"),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "KBCollectionContext", "class"),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "KBContextRequest", "class"),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "KBCoordinator", "class"),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBCoreManagementCompatibilityFacade",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBFileCompatibilityFacade",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "KBHandleProvider", "class"),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBLegacyStepCompatibilityFacade",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "KBMainPointerSnapshot", "class"),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBMaintenanceCompatibilityFacade",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBOperationCompatibilityFacade",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "KBOperationOutcome", "class"),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBParseDisplayCompatibilityFacade",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBPipelineCompatibilityFacade",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBRetrievalHelperCompatibilityFacade",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "KBStorageBackend", "class"),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBStorageShimCompatibilityFacade",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBToolCompatibilityFacade",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "KBUserScope", "class"),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBVectorStorageCleanupResult",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBVectorStorageCompatibilityFacade",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBVersionCandidateCleanupSnapshot",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBVersionCandidateRollbackResult",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "KBVersionCompatibilityFacade",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "LanceDBCollectionHandle",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "PersistencePolicy", "class"),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "RollbackStatus", "class"),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "SideEffectPlane", "class"),
    ("xagent.core.tools.core.RAG_tools.kb.__init__", "get_kb_coordinator", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.kb.__init__",
        "reset_kb_coordinator_for_tests",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.management.__init__",
        "DocumentProcessingStatus",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.management.__init__",
        "cancel_collection",
        "sync",
    ),
    ("xagent.core.tools.core.RAG_tools.management.__init__", "cancel_document", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.management.__init__",
        "clear_ingestion_status",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.management.__init__",
        "clear_ingestion_status_async",
        "async",
    ),
    (
        "xagent.core.tools.core.RAG_tools.management.__init__",
        "delete_collection",
        "sync",
    ),
    ("xagent.core.tools.core.RAG_tools.management.__init__", "delete_document", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.management.__init__",
        "get_document_stats",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.management.__init__",
        "get_document_status",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.management.__init__",
        "list_collections",
        "async",
    ),
    ("xagent.core.tools.core.RAG_tools.management.__init__", "list_documents", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.management.__init__",
        "load_ingestion_status",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.management.__init__",
        "load_ingestion_status_async",
        "async",
    ),
    ("xagent.core.tools.core.RAG_tools.management.__init__", "retry_document", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.management.__init__",
        "write_ingestion_status",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.management.__init__",
        "write_ingestion_status_async",
        "async",
    ),
    ("xagent.core.tools.core.RAG_tools.parse.__init__", "parse_document", "sync"),
    ("xagent.core.tools.core.RAG_tools.pipelines.__init__", "process_document", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.pipelines.__init__",
        "run_web_ingestion",
        "async",
    ),
    ("xagent.core.tools.core.RAG_tools.pipelines.__init__", "search_documents", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.progress.__init__",
        "DeepDocProgressAdapter",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.progress.__init__",
        "FallbackProgressAdapter",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.progress.__init__",
        "ProgressBroadcaster",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.progress.__init__", "ProgressCallback", "class"),
    ("xagent.core.tools.core.RAG_tools.progress.__init__", "ProgressManager", "class"),
    (
        "xagent.core.tools.core.RAG_tools.progress.__init__",
        "ProgressPersistence",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.progress.__init__", "ProgressTracker", "class"),
    ("xagent.core.tools.core.RAG_tools.progress.__init__", "StepTracker", "class"),
    ("xagent.core.tools.core.RAG_tools.progress.__init__", "TaskProgress", "class"),
    (
        "xagent.core.tools.core.RAG_tools.progress.__init__",
        "create_progress_adapter",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.progress.__init__",
        "get_progress_manager",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.progress.__init__",
        "progress_broadcaster",
        "value",
    ),
    (
        "xagent.core.tools.core.RAG_tools.prompt_manager.__init__",
        "create_prompt_template",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.prompt_manager.__init__",
        "delete_prompt_template",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.prompt_manager.__init__",
        "get_latest_prompt_template",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.prompt_manager.__init__",
        "list_prompt_templates",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.prompt_manager.__init__",
        "read_prompt_template",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.prompt_manager.__init__",
        "update_prompt_template",
        "sync",
    ),
    ("xagent.core.tools.core.RAG_tools.retrieval.__init__", "search_dense", "sync"),
    ("xagent.core.tools.core.RAG_tools.retrieval.__init__", "search_hybrid", "sync"),
    ("xagent.core.tools.core.RAG_tools.retrieval.__init__", "search_sparse", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "IngestionStatusStore",
        "class",
    ),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "KBWriteCoordinator",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.storage.__init__", "MainPointerStore", "class"),
    ("xagent.core.tools.core.RAG_tools.storage.__init__", "MetadataStore", "class"),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "PromptTemplateStore",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.storage.__init__", "StorageFactory", "class"),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "VECTOR_BACKEND_ENV",
        "value",
    ),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "VECTOR_BACKEND_ENV_LEGACY",
        "value",
    ),
    ("xagent.core.tools.core.RAG_tools.storage.__init__", "VectorBackend", "class"),
    ("xagent.core.tools.core.RAG_tools.storage.__init__", "VectorIndexStore", "class"),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "get_configured_vector_backend",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "get_ingestion_status_store",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "get_kb_write_coordinator",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "get_main_pointer_store",
        "sync",
    ),
    ("xagent.core.tools.core.RAG_tools.storage.__init__", "get_metadata_store", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "get_prompt_template_store",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "get_vector_index_store",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "get_vector_store_raw_connection",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "reset_kb_write_coordinator",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.storage.__init__",
        "reset_rag_storage_for_tests",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.utils.__init__",
        "build_lancedb_filter_expression",
        "sync",
    ),
    ("xagent.core.tools.core.RAG_tools.utils.__init__", "check_file_type", "sync"),
    ("xagent.core.tools.core.RAG_tools.utils.__init__", "compute_content_hash", "sync"),
    ("xagent.core.tools.core.RAG_tools.utils.__init__", "compute_file_hash", "sync"),
    ("xagent.core.tools.core.RAG_tools.utils.__init__", "deserialize_metadata", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.utils.__init__",
        "escape_lancedb_string",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.utils.__init__",
        "generate_doc_id_from_filename",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.utils.__init__",
        "normalize_raw_embedding_to_vectors",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.utils.__init__",
        "normalize_single_embedding",
        "sync",
    ),
    ("xagent.core.tools.core.RAG_tools.utils.__init__", "query_to_list", "sync"),
    ("xagent.core.tools.core.RAG_tools.utils.__init__", "sanitize_for_doc_id", "sync"),
    ("xagent.core.tools.core.RAG_tools.utils.__init__", "serialize_metadata", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.utils.__init__",
        "validate_and_convert_user_id",
        "sync",
    ),
    ("xagent.core.tools.core.RAG_tools.utils.__init__", "validate_file_path", "sync"),
    ("xagent.core.tools.core.RAG_tools.utils.__init__", "validate_hash_format", "sync"),
    (
        "xagent.core.tools.core.RAG_tools.vector_storage.__init__",
        "commit_vectors_to_db",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.vector_storage.__init__",
        "read_chunks_for_embedding",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.vector_storage.__init__",
        "validate_query_vector",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.vector_storage.__init__",
        "write_vectors_to_db",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.version_management.__init__",
        "cascade_delete",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.version_management.__init__",
        "cleanup_cascade",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.version_management.__init__",
        "cleanup_chunk_cascade",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.version_management.__init__",
        "cleanup_document_cascade",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.version_management.__init__",
        "cleanup_embed_cascade",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.version_management.__init__",
        "cleanup_parse_cascade",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.version_management.__init__",
        "delete_main_pointer",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.version_management.__init__",
        "get_main_pointer",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.version_management.__init__",
        "list_candidates",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.version_management.__init__",
        "list_main_pointers",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.version_management.__init__",
        "promote_version_main",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.version_management.__init__",
        "set_main_pointer",
        "sync",
    ),
    (
        "xagent.core.tools.core.RAG_tools.web_crawler.__init__",
        "ContentCleaner",
        "class",
    ),
    ("xagent.core.tools.core.RAG_tools.web_crawler.__init__", "LinkExtractor", "class"),
    ("xagent.core.tools.core.RAG_tools.web_crawler.__init__", "URLFilter", "class"),
    ("xagent.core.tools.core.RAG_tools.web_crawler.__init__", "WebCrawler", "class"),
    ("xagent.core.tools.core.RAG_tools.web_crawler.__init__", "crawl_website", "async"),
]


EXCLUDED_PUBLIC_SURFACE: dict[tuple[str, str], str] = {
    (
        "xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils",
        "embeddings_table_name",
    ): "module-local table naming helper; no product-code imports outside RAG_tools",
    (
        "xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils",
        "to_model_tag",
    ): "module-local table naming helper; no product-code imports outside RAG_tools",
}


def _public_path(module_path: str) -> str:
    return module_path.removesuffix(".__init__")


def _all_declared_rag_exports() -> set[tuple[str, str]]:
    package_root = Path(rag_tools_pkg.__file__).parent
    package_name = rag_tools_pkg.__name__
    actual: set[tuple[str, str]] = set()
    for py_file in package_root.rglob("*.py"):
        if "__all__" not in py_file.read_text(encoding="utf-8"):
            continue
        relative_module = py_file.relative_to(package_root).with_suffix("")
        parts = relative_module.parts
        if parts == ("__init__",):
            module_name = package_name
        elif parts[-1] == "__init__":
            module_name = ".".join((package_name, *parts[:-1]))
        else:
            module_name = ".".join((package_name, *parts))
        mod = importlib.import_module(module_name)
        exported = getattr(mod, "__all__", None)
        if exported is None:
            continue
        for symbol in exported:
            actual.add((module_name, symbol))
    return actual


@pytest.mark.parametrize(
    "module_path,symbol,kind", PUBLIC_SURFACE, ids=[s for _, s, _ in PUBLIC_SURFACE]
)
def test_public_symbol_importable(module_path, symbol, kind):
    """Every public symbol must remain importable from its declared module path."""
    public_path = _public_path(module_path)
    mod = importlib.import_module(public_path)
    assert hasattr(mod, symbol), f"{symbol} not found in {public_path}"


@pytest.mark.parametrize(
    "module_path,symbol,kind",
    [(m, s, k) for m, s, k in PUBLIC_SURFACE if k in ("sync", "async")],
    ids=[s for m, s, k in PUBLIC_SURFACE if k in ("sync", "async")],
)
def test_public_function_keeps_sync_async_shape(module_path, symbol, kind):
    """Function retained surfaces must keep current sync/async shape.

    Per #507: "Callable retained surfaces keep current sync/async shape."
    """
    public_path = _public_path(module_path)
    mod = importlib.import_module(public_path)
    obj = getattr(mod, symbol)
    runtime_async = inspect.iscoroutinefunction(obj)
    if kind == "async":
        assert runtime_async, f"{symbol} must remain async (currently sync)"
    else:
        assert not runtime_async, f"{symbol} must remain sync (currently async)"


def test_public_surface_completeness():
    """Guard against accidental surface drift.

    Dynamically reads every RAG_tools module's __all__ and compares against the
    hardcoded PUBLIC_SURFACE list plus explicitly documented exclusions. Catches
    both additions and removals, including the case where one symbol is added and
    another removed (net length unchanged).
    """
    expected = {(_public_path(m), s) for m, s, _ in PUBLIC_SURFACE}
    excluded = set(EXCLUDED_PUBLIC_SURFACE)
    actual = _all_declared_rag_exports()

    overlap = expected & excluded
    stale_exclusions = excluded - actual
    un_audited = actual - expected - excluded
    missing = expected - actual
    assert not overlap, (
        f"Symbols cannot be both audited and excluded: {sorted(overlap)}."
    )
    assert not stale_exclusions, (
        f"Excluded symbols are no longer exported in __all__: "
        f"{sorted(stale_exclusions)}. Remove them from EXCLUDED_PUBLIC_SURFACE."
    )
    assert not un_audited, (
        f"Public symbols found in __all__ but not in audit: {sorted(un_audited)}. "
        "Add them to PUBLIC_SURFACE and classify in the audit document."
    )
    assert not missing, (
        f"Audited symbols no longer exported in __all__: {sorted(missing)}. "
        "Remove them from PUBLIC_SURFACE and update the audit document."
    )
