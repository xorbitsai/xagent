import logging
from collections.abc import Sequence

import requests

from ...retry import create_retry_wrapper
from ..model import RerankModelConfig
from .base import BaseRerank

logger = logging.getLogger(__name__)


def retry_on(e: Exception) -> bool:
    ERRORS = requests.exceptions.Timeout

    if isinstance(e, requests.exceptions.HTTPError):
        status_code = e.response.status_code
        return status_code == 429 or 500 <= status_code < 600  # 429 and 5xx
    return isinstance(e, ERRORS)


def _create_rerank_model(model_config: RerankModelConfig) -> BaseRerank:
    """Create the underlying rerank model based on ``model_provider``."""
    provider = (model_config.model_provider or "dashscope").lower()

    if provider == "xinference":
        from .xinference import XinferenceRerank

        return XinferenceRerank(
            model=model_config.model_name,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            top_n=model_config.top_n,
            timeout=model_config.timeout,
        )

    # Default: DashScope-compatible rerank endpoint
    from .dashscope import DashscopeRerank

    return DashscopeRerank(
        model=model_config.model_name,
        api_key=model_config.api_key,
        base_url=model_config.base_url,
        top_n=model_config.top_n,
        instruct=model_config.instruct,
        timeout=model_config.timeout,
    )


def create_rerank_adapter(model_config: RerankModelConfig) -> BaseRerank:
    """
    Creates a custom BaseRerank instance from a RerankModelConfig with retry logic.
    """
    return create_retry_wrapper(
        RerankModelAdapter(model_config),
        BaseRerank,  # type: ignore[type-abstract]
        # Both entry points must retry: production RAG search calls
        # compress_with_scores, so listing only compress silently left the
        # real path with no retry at all.
        retry_methods={"compress", "compress_with_scores"},
        max_retries=model_config.max_retries,
        retry_on=retry_on,
    )


class RerankModelAdapter(BaseRerank):
    """Adapter that makes the new rerank interface compatible with existing RerankModel configs."""

    def __init__(self, model_config: RerankModelConfig):
        self.model_config = model_config
        self._rerank_model = _create_rerank_model(model_config)

    def _record_usage(self, documents: Sequence[str], query: str) -> None:
        """Meter one rerank call; never let accounting break the call itself."""
        try:
            # Lazy import to avoid a circular import via the model package init.
            from ..chat.token_context import (
                MediaCallType,
                MediaUnit,
                add_media_usage,
                estimate_media_tokens,
            )

            # One rerank call is one billable unit regardless of document count,
            # so REQUESTS (always quantity=1) is the correct dimension here.
            texts = list(documents) if documents else []
            if isinstance(query, str):
                texts.append(query)
            add_media_usage(
                unit=MediaUnit.REQUESTS,
                quantity=1,
                model=self.model_config.model_name,
                model_id=self.model_config.id,
                call_type=MediaCallType.RERANK,
                input_tokens=estimate_media_tokens(texts),
                tokens_estimated=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to record rerank usage: %s", e)

    def compress(
        self,
        documents: Sequence[str],
        query: str,
    ) -> Sequence[str]:
        """Rerank documents using the underlying rerank model."""
        result = self._rerank_model.compress(documents, query)
        self._record_usage(documents, query)
        return result

    def compress_with_scores(
        self,
        documents: Sequence[str],
        query: str,
    ) -> list[tuple[str, float]]:
        """Rerank with per-document scores, metering the call.

        The RAG search pipeline needs the scores, so it must be able to get
        them *through* the adapter — reaching the inner provider directly would
        skip metering and leave real rerank usage unbilled.
        """
        result = self._rerank_model.compress_with_scores(documents, query)
        self._record_usage(documents, query)
        return result
