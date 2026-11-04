import logging
from typing import TYPE_CHECKING, List, Optional

from ..core.schemas import SearchResult

if TYPE_CHECKING:
    from ..kb import KBRetrievalHelperCompatibilityFacade

logger = logging.getLogger(__name__)


def format_search_results_for_llm(
    search_results: List[SearchResult],
    top_k: Optional[int] = None,
    include_metadata: bool = False,
    separator: str = "\n---\n",
) -> str:
    """Formats a list of search results into a single string suitable for LLM input.

    This function takes a list of SearchResult objects and converts them into a
    concatenated string. Each SearchResult's 'text' field is extracted.
    Optionally, a subset of top_k results can be selected, and metadata can be
    included in the formatted string.

    Args:
        search_results: A list of SearchResult objects obtained from a search operation.
        top_k: Optional. If provided, only the top_k results will be formatted.
               If None, all results are formatted.
        include_metadata: Optional. If True, includes basic metadata (doc_id, chunk_id, score)
                          for each chunk along with its text. Defaults to False.
        separator: The string used to separate the formatted content of each search result.
                   Defaults to "\\n---\\n".

    Returns:
        A single string containing the formatted content of the search results,
        ready to be used as context for an LLM.
    """
    return _get_retrieval_helper_compatibility_facade().format_search_results_for_llm(
        search_results,
        top_k=top_k,
        include_metadata=include_metadata,
        separator=separator,
    )


def _format_search_results_for_llm_impl(
    search_results: List[SearchResult],
    top_k: Optional[int] = None,
    include_metadata: bool = False,
    separator: str = "\n---\n",
) -> str:
    """Format search results into an LLM context string."""
    if not search_results:
        logger.info("No search results provided for formatting.")
        return ""

    formatted_chunks: List[str] = []
    results_to_format = search_results[:top_k] if top_k is not None else search_results

    for i, result in enumerate(results_to_format):
        chunk_content = result.text
        if include_metadata:
            # Build basic metadata string
            metadata_parts = [
                f"Document ID: {result.doc_id}",
                f"Chunk ID: {result.chunk_id}",
                f"Score: {result.score:.4f}",
            ]

            # Add chunk metadata if available
            if result.metadata:
                metadata_parts.append(f"Metadata: {result.metadata}")

            metadata_str = ", ".join(metadata_parts)
            formatted_chunks.append(f"[{i + 1}]\n{metadata_str}\n{chunk_content}")
        else:
            formatted_chunks.append(f"[{i + 1}]\n{chunk_content}")

    return separator.join(formatted_chunks)


def _get_retrieval_helper_compatibility_facade() -> (
    "KBRetrievalHelperCompatibilityFacade"
):
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().retrieval_helper_compatibility
