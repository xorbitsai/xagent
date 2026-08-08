"""Core document search functionality for RAG pipelines."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ....config import get_kb_search_timeout_seconds
from .RAG_tools.core.schemas import ListCollectionsResult
from .RAG_tools.management.collections import list_collections
from .RAG_tools.pipelines.document_search import run_document_search

logger = logging.getLogger(__name__)

# Prefix of the serialized READONLY_MODE warning that search_sparse raises
# unconditionally under readonly=True (see collection_handle.search_sparse).
# Matched against the string, not SearchWarning.code: the pipeline flattens
# warnings to f"{code}: {message}" in _serialize_warnings before they reach us,
# and the structured object is not plumbed this far. Change that format and this
# filter silently stops matching - the readonly notice reappears in summaries,
# which the readonly tests in test_document_search_collection_concurrency catch.
_READONLY_WARNING_PREFIX = "READONLY_MODE:"

if TYPE_CHECKING:
    from .RAG_tools.kb import KBToolCompatibilityFacade


def _get_tool_compatibility_facade() -> "KBToolCompatibilityFacade":
    """Return the coordinator-owned tool compatibility facade."""
    from .RAG_tools.kb import get_kb_coordinator

    return get_kb_coordinator().tool_compatibility


async def _list_visible_collections(
    user_id: Optional[int], is_admin: bool
) -> ListCollectionsResult:
    """Union personal collections with application-provided team overlays."""
    result = await list_collections(user_id=user_id, is_admin=is_admin)
    if user_id is None or is_admin:
        return result

    from ....web.services.db_runtime import run_db_io_cancellation_safe
    from ....web.services.knowledge_base_team_scope import (
        has_knowledge_base_visibility_hook,
        visible_team_knowledge_bases,
    )

    if not has_knowledge_base_visibility_hook():
        return result

    collections_by_name = {
        collection.name: collection for collection in result.collections
    }
    refs_by_owner: dict[int, list] = {}
    team_refs = await run_db_io_cancellation_safe(
        lambda: visible_team_knowledge_bases(None, int(user_id))
    )
    for ref in team_refs:
        refs_by_owner.setdefault(ref.storage_user_id, []).append(ref)
    for storage_user_id, refs in refs_by_owner.items():
        owner_result = await list_collections(user_id=storage_user_id, is_admin=False)
        owner_collections = {
            collection.name: collection for collection in owner_result.collections
        }
        for ref in refs:
            collection = owner_collections.get(ref.name)
            if collection is None:
                continue
            collections_by_name[ref.name] = collection.model_copy(
                update={
                    "ownership": "team",
                    "storage_user_id": ref.storage_user_id,
                    "can_edit": ref.can_edit,
                    "can_delete": ref.can_delete,
                }
            )
    merged = list(collections_by_name.values())
    return result.model_copy(update={"collections": merged, "total_count": len(merged)})


class ListKnowledgeBasesArgs(BaseModel):
    """Arguments for listing knowledge bases."""

    allowed_collections: Optional[List[str]] = Field(
        default=None,
        description="Optional list of allowed collection names to filter. None means list all collections.",
    )


class ListKnowledgeBasesResult(BaseModel):
    knowledge_bases: List[Dict[str, Any]] = Field(
        description="List of available knowledge bases with statistics"
    )


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(description="The search query or question")
    collections: List[str] = Field(
        default=[],
        description="Specific knowledge base collection names to search. Empty list uses allowed_collections if set, otherwise searches all collections.",
    )
    search_type: str = Field(
        default="hybrid",
        description="Search type: 'dense' (semantic), 'sparse' (keyword), or 'hybrid' (combined)",
    )
    top_k: int = Field(default=5, description="Maximum results per collection")
    min_score: float = Field(
        default=0.3, description="Minimum relevance score (0.0-1.0)"
    )
    embedding_model_id: Optional[str] = Field(
        default=None, description="Optional embedding model ID to use for searches"
    )
    rerank_model_id: Optional[str] = Field(
        default=None,
        description="Optional rerank model ID (registered in model hub) to rerank search results",
    )
    allowed_collections: Optional[List[str]] = Field(
        default=None,
        description="Optional list of allowed collection names. Used as default when collections is empty.",
    )


class SearchResultItem(BaseModel):
    """Single search result with document information."""

    collection: str = Field(description="Knowledge base collection name")
    score: float = Field(description="Relevance score (0.0-1.0)")
    text: str = Field(description="Document text content")
    document_name: str = Field(default="", description="Original document filename")
    source_path: str = Field(default="", description="Full file path")
    doc_id: str = Field(default="", description="Internal document ID")
    chunk_id: str = Field(default="", description="Internal chunk ID")


class KnowledgeSearchResult(BaseModel):
    results: list[SearchResultItem] = Field(
        description="List of search results with document metadata"
    )
    summary: str = Field(
        default="", description="Human-readable summary of search results"
    )


async def list_knowledge_bases(
    tool_args: ListKnowledgeBasesArgs,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> ListKnowledgeBasesResult:
    """List all available knowledge bases through the tool compatibility facade."""
    return await _get_tool_compatibility_facade().list_knowledge_bases(
        tool_args,
        user_id=user_id,
        is_admin=is_admin,
    )


async def _list_knowledge_bases_impl(
    tool_args: ListKnowledgeBasesArgs,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> ListKnowledgeBasesResult:
    """List all available knowledge bases with their statistics.

    Args:
        tool_args: Args with optional allowed_collections filter
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether the user has admin privileges

    Returns:
        ListKnowledgeBasesResult containing knowledge base information

    Raises:
        RuntimeError: If listing knowledge bases fails
    """
    try:
        result = await _list_visible_collections(user_id=user_id, is_admin=is_admin)

        kb_list = []
        for collection in result.collections:
            # Filter by allowed_collections if specified
            if (
                tool_args.allowed_collections is not None
                and collection.name not in tool_args.allowed_collections
            ):
                continue

            kb_list.append(
                {
                    "name": collection.name,
                    "documents": collection.documents,
                    "embeddings": collection.embeddings,
                    "document_names": list(collection.document_names)
                    if collection.document_names
                    else [],
                }
            )

        return ListKnowledgeBasesResult(knowledge_bases=kb_list)

    except Exception as e:
        logger.error(f"Failed to list knowledge bases: {e}", exc_info=True)
        raise RuntimeError(f"Failed to list knowledge bases: {e}") from e


async def find_missing_knowledge_bases(
    knowledge_bases: List[str],
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[str]:
    """Return missing KB names through the tool compatibility facade."""
    return await _get_tool_compatibility_facade().find_missing_knowledge_bases(
        knowledge_bases,
        user_id=user_id,
        is_admin=is_admin,
    )


async def _find_missing_knowledge_bases_impl(
    knowledge_bases: List[str],
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[str]:
    """Return requested knowledge base names that are not visible to the user."""
    requested = [name.strip() for name in knowledge_bases if name and name.strip()]
    if not requested:
        return []

    result = await _list_visible_collections(user_id=user_id, is_admin=is_admin)
    available = {collection.name for collection in result.collections}
    return [name for name in requested if name not in available]


async def search_knowledge_base(
    tool_args: KnowledgeSearchArgs,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> KnowledgeSearchResult:
    """Search across knowledge bases through the tool compatibility facade."""
    return await _get_tool_compatibility_facade().search_knowledge_base(
        tool_args,
        user_id=user_id,
        is_admin=is_admin,
    )


async def _search_knowledge_base_impl(
    tool_args: KnowledgeSearchArgs,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> KnowledgeSearchResult:
    """Search across knowledge base collections.

    Args:
        tool_args: Search configuration including query, collections, and search parameters
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether the user has admin privileges

    Returns:
        KnowledgeSearchResult with formatted search results

    Raises:
        RuntimeError: If search fails
    """
    try:
        # List all collections
        collections_result = await _list_visible_collections(
            user_id=user_id, is_admin=is_admin
        )

        if not collections_result.collections:
            return KnowledgeSearchResult(
                results=[],
                summary="No knowledge bases available. Please create a knowledge base and upload documents first.",
            )

        # Determine which collections to search
        available_names = {c.name for c in collections_result.collections}

        # Debug: Log available collections for troubleshooting
        logger.info(
            f"📚 Available knowledge base collections: {sorted(available_names)}"
        )
        if tool_args.collections:
            logger.info(f"   - Requested collections: {tool_args.collections}")
        if tool_args.allowed_collections:
            logger.info(f"   - Allowed collections: {tool_args.allowed_collections}")

        if tool_args.collections:
            # User specified collections - validate against allowed_collections
            requested_set = set(tool_args.collections)

            # If allowed_collections is set, verify requested is a subset
            if tool_args.allowed_collections is not None:
                allowed_set = set(tool_args.allowed_collections)
                disallowed = requested_set - allowed_set

                if disallowed:
                    return KnowledgeSearchResult(
                        results=[],
                        summary=f"Error: The following collections are not allowed: {', '.join(sorted(disallowed))}. "
                        f"Allowed collections: {', '.join(sorted(allowed_set & available_names))}",
                    )

                collections_set = requested_set & allowed_set
            else:
                collections_set = requested_set

            # Check if collections exist
            invalid_names = collections_set - available_names
            if invalid_names:
                return KnowledgeSearchResult(
                    results=[],
                    summary=f"Error: The following collections do not exist: {', '.join(invalid_names)}. "
                    f"Available collections: {', '.join(sorted(available_names))}",
                )

            collections_to_iterate = [
                c for c in collections_result.collections if c.name in collections_set
            ]
            logger.info(f"Searching specific collections: {sorted(collections_set)}")
        elif tool_args.allowed_collections is not None:
            # Use allowed_collections as default
            allowed_set = set(tool_args.allowed_collections)

            if not allowed_set:
                return KnowledgeSearchResult(
                    results=[],
                    summary="Knowledge base search is disabled for this agent (no knowledge bases configured).",
                )
            valid_collections = allowed_set & available_names

            if not valid_collections:
                return KnowledgeSearchResult(
                    results=[],
                    summary=f"Error: None of the allowed collections exist. "
                    f"Allowed: {', '.join(sorted(allowed_set))}. "
                    f"Available: {', '.join(sorted(available_names))}",
                )

            collections_to_iterate = [
                c for c in collections_result.collections if c.name in valid_collections
            ]
            logger.info(f"Searching allowed collections: {sorted(valid_collections)}")
        else:
            collections_to_iterate = collections_result.collections
            logger.info("Searching all collections")

        # Build base search config (per-collection overrides happen below)
        base_search_config = {
            "search_type": tool_args.search_type,
            "top_k": tool_args.top_k,
            "min_score": tool_args.min_score,
            "merge_results": True,
            # Retrieval must not *build indexes*: create_index() commits to the
            # LanceDB table, and searching collections concurrently can now race
            # that commit (see collection_manager's CommitConflict note).
            # Indexes are built during ingestion, which is where they belong.
            #
            # This suppresses both index paths, not just FTS: the readonly branch
            # of create_index returns before the dense/vector block and before the
            # FTS block (lancedb_stores.py). Only the sparse path reports it
            # (READONLY_MODE / FTS_INDEX_MISSING) - dense search hardcodes
            # warnings=[] on success, so a skipped dense index build is silent.
            #
            # Not a claim that the search writes nothing: resolving the embedding
            # model still stamps last_accessed_at per collection
            # (collection_manager.mark_collection_accessed). That row-overwrite is
            # pre-existing and guarded by a per-collection lock, so the N
            # concurrent searches this fan-out creates land on N distinct keys.
            "readonly": True,
        }

        if tool_args.embedding_model_id:
            base_search_config["embedding_model_id"] = tool_args.embedding_model_id

        # Search across collections and aggregate results
        all_results = []
        collection_errors: list[str] = []
        collection_warnings: list[str] = []
        total_searched = 0
        search_timeout_seconds = get_kb_search_timeout_seconds()

        async def _search_one(
            collection_info: Any,
        ) -> tuple[list[Dict[str, Any]], Optional[str], Optional[str], int]:
            """Search one collection off the event loop.

            Returns (results, error, warning, documents_searched); failures are
            returned rather than raised so one collection cannot fail the batch.
            """
            # Every attribute read lives inside the try below, so this really
            # cannot raise into the gather. _failure reads the name lazily, so
            # it stays usable even if that first read is what failed.
            collection_name = "<unknown>"

            def _failure(
                reason: str,
            ) -> tuple[list[Dict[str, Any]], Optional[str], Optional[str], int]:
                return [], f"{collection_name}: {reason}", None, 0

            try:
                collection_name = collection_info.name

                # Per-KB rerank resolution: explicit tool arg wins, otherwise
                # use the collection's bound rerank_model_id; when neither is
                # set, no rerank stage is added for this collection.
                search_config = dict(base_search_config)
                collection_rerank = getattr(collection_info, "rerank_model_id", None)
                effective_rerank = tool_args.rerank_model_id or collection_rerank
                if effective_rerank:
                    search_config["rerank_model_id"] = effective_rerank
                storage_user_id = getattr(collection_info, "storage_user_id", None)

                logger.info(
                    f"Searching collection '{collection_name}' for: {tool_args.query}"
                )

                # run_document_search is a blocking sync pipeline; running it
                # inline would pin the event loop for the whole retrieval.
                # The deadline covers queueing too, not just execution: it starts
                # when this coroutine is scheduled, so a saturated default
                # executor can burn it before run_document_search even starts.
                # And it frees this caller but not the worker - a timed-out
                # to_thread call keeps running in that shared executor.
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_document_search,
                        collection=collection_name,
                        query_text=tool_args.query,
                        config=search_config,
                        user_id=storage_user_id
                        if storage_user_id is not None
                        else user_id,
                        is_admin=False
                        if getattr(collection_info, "ownership", "personal") == "team"
                        else is_admin,
                    ),
                    timeout=search_timeout_seconds,
                )

                if result.status not in {"success", "partial_success"}:
                    error_message = result.message or "; ".join(result.warnings)
                    logger.warning(
                        "Search pipeline returned status '%s' for collection '%s': %s",
                        result.status,
                        collection_name,
                        error_message,
                    )
                    return _failure(f"{error_message or 'search failed'}")

                # The pipeline's message on a non-success status is boilerplate
                # ("Hybrid search completed with warnings"), so the warning list
                # has to win or every real diagnostic is masked by it. The
                # readonly notice is self-inflicted by our own readonly=True and
                # fires on every search; FTS_INDEX_MISSING, the warning that
                # reports an actual consequence of it, is kept.
                warning: Optional[str] = None
                warning_message = "; ".join(
                    w
                    for w in result.warnings
                    if not w.startswith(_READONLY_WARNING_PREFIX)
                )
                if warning_message:
                    warning = f"{collection_name}: {warning_message}"

                if not result.results:
                    return [], None, warning, 0

                results = []
                for res in result.results:
                    res_dict = dict(res)
                    res_dict["collection"] = collection_name
                    results.append(res_dict)
                return results, None, warning, collection_info.documents

            except asyncio.TimeoutError:
                logger.warning(
                    "Search of collection '%s' exceeded %ss",
                    collection_name,
                    search_timeout_seconds,
                )
                return _failure(f"search timed out after {search_timeout_seconds}s")
            except Exception as e:
                logger.warning(f"Failed to search collection '{collection_name}': {e}")
                return _failure(str(e))

        # Skip collections with no embeddings before fanning out.
        searchable = []
        for collection_info in collections_to_iterate:
            if collection_info.embeddings == 0:
                logger.debug(
                    f"Skipping collection with no embeddings: {collection_info.name}"
                )
                continue
            searchable.append(collection_info)

        # Search every collection concurrently: total latency is bounded by the
        # slowest collection instead of the sum of all of them. gather preserves
        # input order, so aggregation order is unchanged.
        # ponytail: no concurrency cap. Holds for the bound-KB paths above, where
        # the agent's own configuration is the bound. It does NOT hold for the
        # final fallback branch: with neither `collections` nor
        # `allowed_collections` set, this fans out over every collection visible
        # to the caller, which _list_visible_collections unions across owners -
        # unbounded by anything the agent declared. Add a Semaphore when that
        # branch gets real traffic, or when fan-out width starves the shared
        # default executor that every asyncio.to_thread caller draws from.
        for results, error, warning, documents in await asyncio.gather(
            *(_search_one(collection_info) for collection_info in searchable)
        ):
            if error:
                collection_errors.append(error)
                continue
            if warning:
                collection_warnings.append(warning)
            if results:
                all_results.extend(results)
                total_searched += documents

        if not all_results:
            if collection_errors:
                summary = (
                    "Knowledge base search failed for one or more collections: "
                    + " | ".join(collection_errors)
                )
                if collection_warnings:
                    summary = (
                        summary + "\n\nWarnings: " + " | ".join(collection_warnings)
                    )
                return KnowledgeSearchResult(results=[], summary=summary)
            summary = (
                f"No relevant documents found in any knowledge base. "
                f"Searched {total_searched} documents across "
                f"{len(collections_result.collections)} collections. Query: {tool_args.query}"
            )
            if collection_warnings:
                summary = summary + "\n\nWarnings: " + " | ".join(collection_warnings)
            return KnowledgeSearchResult(results=[], summary=summary)

        # Format results (structured + summary)
        formatted_results, summary = _format_search_results(
            all_results, tool_args.query, total_searched
        )
        if collection_warnings:
            summary = summary + "\n\nWarnings: " + " | ".join(collection_warnings)
        if collection_errors:
            summary = summary + "\n\nErrors: " + " | ".join(collection_errors)

        return KnowledgeSearchResult(results=formatted_results, summary=summary)

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        raise RuntimeError(f"Knowledge base search failed: {e}") from e


def _format_search_results(
    results: List[Dict[str, Any]], query: str, total_documents: int
) -> tuple[list[Dict[str, Any]], str]:
    """Format search results for LLM consumption.

    Returns:
        Tuple of (structured_results, summary_string)
    """
    formatted_results = []

    for result in results:
        collection = result.get("collection", "unknown")
        score = result.get("score", 0.0)
        text = result.get("text", "")
        metadata = result.get("metadata") or {}

        # Extract file information from metadata
        source_path = metadata.get("source", "")
        doc_id = metadata.get("doc_id", "")
        chunk_id = metadata.get("chunk_id", "")

        # Try to get document name from source_path
        document_name = ""
        if source_path:
            import os

            document_name = os.path.basename(source_path)

        # Create structured result
        structured_result = {
            "collection": collection,
            "score": score,
            "text": text,
            "document_name": document_name,
            "source_path": source_path,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
        }
        formatted_results.append(structured_result)

    # Create brief summary (token-efficient, no duplicate content)
    summary = f"Found {len(results)} relevant results from {total_documents} documents for query: '{query}'"

    return formatted_results, summary
