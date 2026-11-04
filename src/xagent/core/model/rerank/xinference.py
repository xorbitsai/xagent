"""Xinference Rerank provider implementation.

Uses Xinference's OpenAI-compatible ``/v1/rerank`` HTTP endpoint, so the
implementation does not need a long-lived model handle. This matches the
behaviour of ``xinference_embedding`` (which lazily obtains a handle) and
keeps request lifecycle aligned with ``DashscopeRerank`` (sync ``requests``).

Reference: https://inference.readthedocs.io/en/latest/models/builtin/rerank.html
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

from .. import public_requests as requests
from ..retrieval_metering import read_billed_json, record_rerank_usage, reported_tokens
from .base import BaseRerank

logger = logging.getLogger(__name__)


class XinferenceRerank(BaseRerank):
    """Xinference rerank model implementation (OpenAI-compatible ``/v1/rerank``)."""

    def __init__(
        self,
        model: str,
        model_uid: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        top_n: Optional[int] = None,
        timeout: Optional[float] = None,
        model_id: str = "",
    ):
        """
        Initialize Xinference rerank model.

        Args:
            model: Model name (e.g., ``bge-reranker-base``, ``Qwen3-Reranker``).
                Used as the ``model`` field in the rerank request when
                ``model_uid`` is not provided.
            model_uid: Unique model UID in Xinference (if the model is already
                launched). When provided, this value is sent as the ``model``
                field of the request.
            api_key: Optional API key for authentication. Falls back to the
                ``XINFERENCE_API_KEY`` environment variable.
            base_url: Xinference server base URL, e.g.
                ``http://localhost:9997``. Defaults to
                ``http://localhost:9997``.
            top_n: Optional cap on the number of reranked results to return.
            timeout: HTTP request timeout in seconds (default: 60).
            model_id: Configured model id recorded on usage rows.
        """
        self.model = model
        self.model_id = model_id
        self._model_uid = model_uid or model
        self.api_key = api_key or os.getenv("XINFERENCE_API_KEY")
        self.base_url = (base_url or "http://localhost:9997").rstrip("/")
        self.top_n = top_n
        self.timeout = float(timeout) if timeout is not None else 60.0

    def _post(self, documents: list[str], query: str) -> Any:
        """Send one rerank request and return its parsed body.

        Records the request as soon as it is accepted, before the body is
        read. ``return_len`` asks the server to report token counts in
        ``meta.tokens``; its ``output_tokens`` repeats the input count (a
        rerank produces scores, not tokens), so only the input is recorded.
        """
        url = f"{self.base_url}/v1/rerank"
        payload: dict[str, Any] = {
            "model": self._model_uid,
            "query": query,
            "documents": documents,
            "return_len": True,
        }
        if self.top_n is not None:
            payload["top_n"] = self.top_n

        logger.debug(
            "Xinference rerank request: url=%s model=%s n_docs=%d top_n=%s",
            url,
            self._model_uid,
            len(documents),
            self.top_n,
        )

        response = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        try:
            return read_billed_json(
                response,
                lambda body: record_rerank_usage(
                    self, reported_tokens(body, ("meta", "tokens", "input_tokens"))
                ),
            )
        except ValueError as exc:
            raise ValueError(
                f"Xinference rerank endpoint returned non-JSON response: "
                f"{response.text[:200]!r}"
            ) from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def list_available_models(
        base_url: str, api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch available rerank models from a Xinference server.

        Uses the synchronous ``xinference_client`` to query the server. The
        server's ``list_models`` returns a mapping keyed by ``model_uid``;
        entries with ``model_type == "rerank"`` are returned with a stable
        shape compatible with the model-management UI.

        Args:
            base_url: Xinference server base URL.
            api_key: Optional API key for authentication.

        Returns:
            List of available rerank models, each shaped as
            ``{"id", "model_uid", "model_type", "model_ability", "description"}``.

        Example:
            >>> models = XinferenceRerank.list_available_models(
            ...     base_url="http://localhost:9997"
            ... )
        """
        from ..public_xinference import PublicXinferenceClient as XinferenceClient

        client = XinferenceClient(base_url=base_url.rstrip("/"), api_key=api_key)
        try:
            models_dict = client.list_models()
        except Exception as exc:
            logger.error("Failed to fetch rerank models from Xinference: %s", exc)
            return []
        finally:
            try:
                client.close()
            except Exception:  # pragma: no cover
                pass

        result: List[Dict[str, Any]] = []
        for model_uid, model_info in (models_dict or {}).items():
            if not isinstance(model_info, dict):
                continue
            if model_info.get("model_type") != "rerank":
                continue
            result.append(
                {
                    "id": model_info.get("model_name", model_uid),
                    "model_uid": model_uid,
                    "model_type": model_info.get("model_type", ""),
                    "model_ability": model_info.get("model_ability", []),
                    "description": model_info.get("model_description", ""),
                }
            )
        return result

    def compress_with_scores(
        self,
        documents: Sequence[str],
        query: str,
    ) -> list[tuple[str, float]]:
        """Rerank documents, returning (text, relevance_score) tuples.

        This variant preserves the rerank model's relevance score so the
        downstream pipeline can overwrite SearchResult.score with a value
        that reflects the rerank stage (instead of the original
        embedding/RRF score).

        Args:
            documents: Documents to rerank.
            query: Query string.

        Returns:
            List of (text, relevance_score) tuples, ordered by descending
            relevance. If the response contains fewer entries than the input,
            the remaining items are appended in their original order with
            score=0.0.

        Raises:
            requests.HTTPError: If the API call returns a non-2xx status.
            ValueError: If the response cannot be parsed.
        """
        documents = list(documents)
        if not documents:
            return []

        data = self._post(documents, query)

        results = data.get("results")
        if not isinstance(results, list):
            raise ValueError(
                f"Xinference rerank response missing 'results' list: "
                f"{list(data.keys())}"
            )

        ordered_pairs: list[tuple[str, float]] = []
        seen: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                logger.warning(
                    "Xinference rerank: skipping non-dict result entry: %r", item
                )
                continue
            try:
                index = int(item["index"])
                score = float(item["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Xinference rerank result missing/invalid 'index' or "
                    f"'relevance_score': {item}"
                ) from exc
            if 0 <= index < len(documents) and index not in seen:
                ordered_pairs.append((documents[index], score))
                seen.add(index)
            else:
                logger.warning(
                    "Xinference rerank: skipping out-of-range or duplicate "
                    "index %s (n_docs=%d)",
                    index,
                    len(documents),
                )

        # Preserve any documents the reranker dropped (e.g. when top_n is
        # smaller than the input size). They get a score of 0.0 so they
        # appear at the bottom.
        for idx, doc in enumerate(documents):
            if idx not in seen:
                ordered_pairs.append((doc, 0.0))

        return ordered_pairs

    def compress(
        self,
        documents: Sequence[str],
        query: str,
    ) -> Sequence[str]:
        """
        Rerank documents by relevance to the query.

        Uses Xinference's OpenAI-compatible rerank endpoint. The response is
        expected to be::

            {
                "results": [
                    {"index": 0, "relevance_score": 0.97},
                    ...
                ]
            }

        Args:
            documents: Documents to rerank.
            query: Query string.

        Returns:
            Documents reordered by descending relevance. If the response
            contains fewer entries than the input, the remaining items are
            appended in their original order to preserve content.

        Raises:
            requests.HTTPError: If the API call returns a non-2xx status.
            ValueError: If the response cannot be parsed.
        """
        documents = list(documents)
        if not documents:
            return []

        data = self._post(documents, query)

        results = data.get("results")
        if not isinstance(results, list):
            raise ValueError(
                f"Xinference rerank response missing 'results' list: "
                f"{list(data.keys())}"
            )

        ordered: list[str] = []
        seen: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                logger.warning(
                    "Xinference rerank: skipping non-dict result entry: %r", item
                )
                continue
            try:
                index = int(item["index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Xinference rerank result missing/invalid 'index': {item}"
                ) from exc
            if 0 <= index < len(documents) and index not in seen:
                ordered.append(documents[index])
                seen.add(index)
            else:
                logger.warning(
                    "Xinference rerank: skipping out-of-range or duplicate "
                    "index %s (n_docs=%d)",
                    index,
                    len(documents),
                )

        # Preserve any documents the reranker dropped (e.g. when top_n is
        # smaller than the input size).
        for idx, doc in enumerate(documents):
            if idx not in seen:
                ordered.append(doc)

        return ordered
