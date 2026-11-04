"""Usage recording at the provider boundary for embedding and rerank.

Same rules as ``media_metering`` for the other non-image modalities:

1. **One row per billed HTTP call.** Recorded inside the provider method that
   made the call. A retry wrapper above it records once per attempt that the
   provider accepted; a failed attempt (HTTP error, timeout) records nothing.
   DashScope embedding splits a request into sub-batches of at most ten texts,
   each its own billed call, so each sub-batch is its own row.
2. **Recorded once the provider has accepted the call, before the response is
   validated.** A 2xx body that fails to parse, or parses into the wrong
   shape, still leaves its row; it just carries no provider tokens.
3. **Provider tokens are never estimated.** They are recorded only when the
   response reports them. Embedding rows then also carry the text/image input
   split (all text), so a per-token price can be applied in full.
4. **Identity is the configured row.** ``model_id`` is stamped on the provider
   at construction, ``model`` is the provider-facing name. Placeholders are
   written as "" and never billed.

Recording is best-effort: nothing in this module raises into the call it
measures. Its imports of the usage machinery are deferred: the embedding
package is the first thing ``xagent.core.model`` imports, before the chat
package that machinery lives in can be initialised.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

_MAX_REPORTED_TOKENS = 1_000_000_000


def _field(container: Any, name: str) -> Any:
    """``container[name]`` or ``container.name``; None when absent or unreadable."""
    try:
        if isinstance(container, Mapping):
            return container.get(name)
        return getattr(container, name, None)
    except Exception:  # noqa: BLE001
        return None


def reported_tokens(container: Any, *path_options: tuple[str, ...]) -> Optional[int]:
    """The first usable token count found along any of ``path_options``.

    Each option is a key path into ``container`` (dicts or SDK objects).
    Booleans, negatives, non-integers and absurd values are not a report;
    ``None`` means the provider reported nothing usable.
    """
    for path in path_options:
        value: Any = container
        for name in path:
            value = _field(value, name)
            if value is None:
                break
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) and not value.is_integer():
            continue
        tokens = int(value)
        if 0 <= tokens <= _MAX_REPORTED_TOKENS:
            return tokens
    return None


def record_embedding_usage(
    provider: Any, texts: Any, input_tokens: Optional[int]
) -> None:
    """Record one billed embedding call of ``texts`` texts; never raises."""
    try:
        from .chat.token_context import MediaCallType, add_media_usage
        from .media_metering import provider_identity

        model, model_id = provider_identity(provider)
        token_fields: dict[str, Any] = {}
        if input_tokens is not None:
            # An embedding request is text only, so the whole input is the
            # text portion. Recording the split lets a per-token price apply
            # to the complete row instead of reading it as partial coverage.
            token_fields = {
                "input_tokens": input_tokens,
                "provider_text_input_tokens": input_tokens,
                "provider_image_input_tokens": 0,
            }
        add_media_usage(
            call_type=MediaCallType.EMBEDDING,
            quantity=texts,
            model=model,
            model_id=model_id,
            **token_fields,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to record embedding usage: %s", e, exc_info=True)


def record_rerank_usage(provider: Any, input_tokens: Optional[int]) -> None:
    """Record one billed rerank request; never raises."""
    try:
        from .chat.token_context import MediaCallType, add_media_usage
        from .media_metering import provider_identity

        model, model_id = provider_identity(provider)
        add_media_usage(
            call_type=MediaCallType.RERANK,
            quantity=1,
            model=model,
            model_id=model_id,
            input_tokens=input_tokens or 0,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to record rerank usage: %s", e, exc_info=True)


def read_billed_json(response: Any, record: Callable[[Any], None]) -> Any:
    """Parse an accepted response, recording the billed call exactly once.

    ``record`` receives the parsed body, or None when the body does not parse
    -- the provider charged for the call either way, so the parse error is
    raised only after the row is written.
    """
    try:
        data = response.json()
    except BaseException:
        record(None)
        raise
    record(data)
    return data
