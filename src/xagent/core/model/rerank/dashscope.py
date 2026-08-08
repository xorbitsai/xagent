import os
from collections.abc import Sequence
from typing import Any, Optional

import requests

from .base import BaseRerank

# Model names that use the OpenAI-compatible /compatible-api/v1/reranks
# endpoint with the new payload/response shape.
NEW_FORMAT_MODELS = {"qwen3-rerank"}
# Model names that use the legacy WebAPI endpoint
# (/api/v1/services/rerank/text-rerank/text-rerank) with the
# `input`/`parameters` payload and `output.results` response.
OLD_FORMAT_MODELS = {"gte-rerank-v2", "qwen3-vl-rerank"}

OLD_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
NEW_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
"""Default endpoints, keyed by format family.

See: https://help.aliyun.com/zh/model-studio/text-rerank-api
"""


def _default_url_for(model: str) -> str:
    """Return the default endpoint for ``model``."""
    if model.lower() in NEW_FORMAT_MODELS:
        return NEW_URL
    return OLD_URL


class DashscopeRerank(BaseRerank):
    """Dashscope rerank model implementation.

    Supports both API formats documented at
    https://help.aliyun.com/zh/model-studio/text-rerank-api:

    * New format (``qwen3-rerank``): OpenAI-compatible
      ``/compatible-api/v1/reranks`` endpoint, with top-level
      ``query``/``documents``/``top_n``/``instruct`` and a flat
      ``results`` array.
    * Old format (``gte-rerank-v2``, ``qwen3-vl-rerank``): legacy WebAPI
      ``/api/v1/services/rerank/text-rerank/text-rerank`` endpoint, with
      ``input``/``parameters`` and an ``output.results`` array that
      echoes the document text.
    """

    def __init__(
        self,
        model: str = "qwen3-rerank",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        top_n: Optional[int] = None,
        instruct: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize Dashscope rerank model.

        Args:
            model: Model name (default: ``qwen3-rerank``).
            api_key: API key (defaults to ``DASHSCOPE_API_KEY`` env var).
            base_url: API base URL. Defaults to the endpoint matching
                ``model``'s format family.
            top_n: Number of top results to return.
            instruct: Custom instruction for reranking (new-format models).
            timeout: HTTP request timeout in seconds (default: 60).
        """
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.top_n = top_n
        self.instruct = instruct
        self.timeout = float(timeout) if timeout is not None else 60.0
        self.url = base_url or _default_url_for(model)

        if not self.api_key:
            raise ValueError("API key required")

    @property
    def is_new_format(self) -> bool:
        return self.model.lower() in NEW_FORMAT_MODELS

    def compress(
        self,
        documents: Sequence[str],
        query: str,
    ) -> Sequence[str]:
        """
        Rerank documents based on query relevance.

        Args:
            documents: List of document strings to rerank.
            query: Query string.

        Returns:
            Reranked list of documents.

        Raises:
            requests.HTTPError: If API returns non-2xx status.
            KeyError: If API response has unexpected format.
            ValueError: If index in response is invalid.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        documents = list(documents)

        if not documents:
            return []

        optional_params: dict[str, Any] = {}
        if self.top_n is not None:
            optional_params["top_n"] = self.top_n
        if self.instruct is not None:
            optional_params["instruct"] = self.instruct

        payload: dict[str, Any]
        if self.is_new_format:
            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
            } | optional_params
        else:
            payload = {
                "model": self.model,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {"return_documents": True} | optional_params,
            }

        response = requests.post(
            self.url, headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        if self.is_new_format:
            # New qwen3-rerank format: no nested "output" wrapper, no
            # "document.text" in results.
            # eg:
            # {"object":"list","results":[{"index":0,"relevance_score":0.92}, ...],
            #  "model":"qwen3-rerank","id":"...","usage":{"total_tokens":105}}
            results = data["results"]
            return [documents[int(result["index"])] for result in results]

        # Old format: gte-rerank-v2 / qwen3-vl-rerank.
        # eg:
        # {"output":{"results":[{"document":{"text":"..."},"index":0,
        #                       "relevance_score":0.92}, ...]},
        #  "usage":{...},"request_id":"..."}
        results = data["output"]["results"]
        return [result["document"]["text"] for result in results]

    def compress_with_scores(
        self,
        documents: Sequence[str],
        query: str,
    ) -> list[tuple[str, float]]:
        """Like :meth:`compress` but also returns the relevance score per doc.

        Returns a list of ``(text, relevance_score)`` tuples ordered by
        descending relevance.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        documents = list(documents)
        if not documents:
            return []

        optional_params: dict[str, Any] = {}
        if self.top_n is not None:
            optional_params["top_n"] = self.top_n
        if self.instruct is not None:
            optional_params["instruct"] = self.instruct

        payload: dict[str, Any]
        if self.is_new_format:
            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
            } | optional_params
        else:
            payload = {
                "model": self.model,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {"return_documents": True} | optional_params,
            }

        response = requests.post(
            self.url, headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        if self.is_new_format:
            results = data["results"]
            return [
                (
                    documents[int(item["index"])],
                    float(item.get("relevance_score", 0.0)),
                )
                for item in results
            ]

        results = data["output"]["results"]
        return [
            (
                item["document"]["text"],
                float(item.get("relevance_score", 0.0)),
            )
            for item in results
        ]
