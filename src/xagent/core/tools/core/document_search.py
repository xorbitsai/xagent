"""Core document search functionality for RAG pipelines."""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .RAG_tools.management.collections import list_collections
from .RAG_tools.pipelines.document_search import run_document_search

logger = logging.getLogger(__name__)


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
        result = await list_collections(user_id=user_id, is_admin=is_admin)

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


async def search_knowledge_base(
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
        collections_result = await list_collections(user_id=user_id, is_admin=is_admin)

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
        elif tool_args.allowed_collections:
            # Use allowed_collections as default
            allowed_set = set(tool_args.allowed_collections)
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

        # Build search config
        search_config = {
            "search_type": tool_args.search_type,
            "top_k": tool_args.top_k,
            "min_score": tool_args.min_score,
            "merge_results": True,
        }

        if tool_args.embedding_model_id:
            search_config["embedding_model_id"] = tool_args.embedding_model_id

        # Search across collections and aggregate results
        all_results = []
        total_searched = 0

        for collection_info in collections_to_iterate:
            collection_name = collection_info.name

            # Skip collections with no embeddings
            if collection_info.embeddings == 0:
                logger.debug(
                    f"Skipping collection with no embeddings: {collection_name}"
                )
                continue

            try:
                logger.info(
                    f"Searching collection '{collection_name}' for: {tool_args.query}"
                )

                result = run_document_search(
                    collection=collection_name,
                    query_text=tool_args.query,
                    config=search_config,
                    user_id=user_id,
                    is_admin=is_admin,
                )

                if result.results:
                    for res in result.results:
                        res_dict = dict(res)
                        res_dict["collection"] = collection_name
                        all_results.append(res_dict)

                    total_searched += collection_info.documents

            except Exception as e:
                logger.warning(f"Failed to search collection '{collection_name}': {e}")
                continue

        if not all_results:
            return KnowledgeSearchResult(
                results=[],
                summary=f"No relevant documents found in any knowledge base. "
                f"Searched {total_searched} documents across "
                f"{len(collections_result.collections)} collections. Query: {tool_args.query}",
            )

        # Format results (structured + summary)
        formatted_results, summary = _format_search_results(
            all_results, tool_args.query, total_searched
        )

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
        metadata = result.get("metadata", {})

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
