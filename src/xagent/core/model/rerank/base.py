from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseRerank(ABC):
    """Abstract base class for rerank models."""

    @abstractmethod
    def compress(
        self,
        documents: Sequence[str],
        query: str,
    ) -> Sequence[str]:
        """
        Rerank documents by relevance to the query.

        Args:
            documents: Candidate documents to rerank
            query: Query to score the documents against

        Returns:
            The documents ordered by descending relevance.
        """
        pass

    def compress_with_scores(
        self,
        documents: Sequence[str],
        query: str,
    ) -> list[tuple[str, float]]:
        """Like :meth:`compress` but also returns the relevance score per doc.

        Declared on the base — not only on the concrete providers — so callers
        can reach it through the adapter. Reaching it by unwrapping the
        adapter's inner provider skips usage metering, which is why the RAG
        search pipeline previously recorded no rerank usage at all.

        Returns ``(text, relevance_score)`` tuples ordered by descending
        relevance. The default derives descending pseudo-scores from
        :meth:`compress`'s ordering rather than returning all-zero scores: a
        caller that writes these into a result's ``score`` field would
        otherwise silently flatten every result to 0.0 while reporting that
        reranking succeeded. Ordering is still authoritative; the magnitudes
        are synthetic, so providers with real relevance scores should override
        this.
        """
        ordered = list(self.compress(documents, query))
        if not ordered:
            return []
        # Evenly spaced in (0, 1], preserving the provider's ordering.
        step = 1.0 / len(ordered)
        return [(text, 1.0 - index * step) for index, text in enumerate(ordered)]
