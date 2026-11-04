"""Configuration utility functions for RAG pipelines."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

from ..core.schemas import IngestionConfig, SearchConfig

IngestionConfigInput = Union[IngestionConfig, Mapping[str, Any]]
SearchConfigInput = Union[SearchConfig, Mapping[str, Any]]


def coerce_ingestion_config(config: Optional[IngestionConfigInput]) -> IngestionConfig:
    """Normalize user-provided ingestion configuration into ``IngestionConfig``."""

    if config is None:
        return IngestionConfig()
    if isinstance(config, IngestionConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError(
            "ingestion_config must be an IngestionConfig instance or a mapping."
        )
    return IngestionConfig.model_validate(config)


def coerce_search_config(search_config: SearchConfigInput) -> SearchConfig:
    """Normalize arbitrary search configuration input into ``SearchConfig``."""

    if isinstance(search_config, SearchConfig):
        return search_config
    if not isinstance(search_config, Mapping):
        raise TypeError(
            "search_config must be a SearchConfig instance or a mapping of fields."
        )
    payload = dict(search_config)
    # Handle embedding_model_id: preserve "default" and "none" for resolver to handle
    # Resolver logic:
    # - "default" -> try hub.load("default"), fallback to env
    # - "none" or empty -> try hub.load("default"), fallback to env
    embedding_model_id = payload.get("embedding_model_id")

    if embedding_model_id and isinstance(embedding_model_id, str):
        # Normalize case for comparison, but preserve original value
        normalized = embedding_model_id.strip().lower()
        if normalized == "none" or normalized == "":
            # Convert "none" or empty to "none" (standardized) for resolver
            payload["embedding_model_id"] = "none"
        elif normalized == "default":
            # Preserve "default" as-is for resolver
            payload["embedding_model_id"] = "default"
        # Otherwise, keep original value (will be validated by Pydantic)
    else:
        # If missing or not a string, set to "none" to trigger resolver's default lookup logic
        payload["embedding_model_id"] = "none"

    return SearchConfig.model_validate(payload)
