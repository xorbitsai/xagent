"""Switch one collection's bound embedding model while it is frozen.

The switch writes the model id and dimension together, records the transition in
``collection_migration_state``, and leaves every other metadata column and
``main_pointers`` untouched. Rolling back is the same call with the previous
model, after the collection is frozen again, provided that model is still
configured under the id it is recorded with: a legacy row bound to an alias that
no longer resolves to itself cannot be rolled back by this call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from xagent.core.model.model import EmbeddingModelConfig

from ..core.exceptions import CollectionNotFrozenError, EmbeddingAdapterError
from ..utils.model_resolver import _load_model_from_hub
from .collection_manager import (
    _normalize_model_id,
    get_collection_sync,
    write_collection_binding_sync,
)
from .migration_state import (
    FROZEN,
    CollectionMigrationState,
    MigrationWindow,
    read_migration_state,
    record_migration_transition,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectionEmbeddingSwitch:
    """One applied switch: what the collection used and what it uses now."""

    collection_name: str
    source_model_id: Optional[str]
    target_model_id: str
    target_dimension: int


def _resolve_configured_model(
    target_model_id: str,
) -> tuple[str, int, EmbeddingModelConfig]:
    """Return the hub config for ``target_model_id``, never an env fallback.

    Binding a model the hub does not know would silently switch the collection to
    whatever the environment default is.
    """
    config = _load_model_from_hub(
        target_model_id, EmbeddingModelConfig, EmbeddingAdapterError
    )
    canonical = _normalize_model_id(config.id)
    if not canonical or canonical != _normalize_model_id(target_model_id):
        raise ValueError(
            f"Embedding model '{target_model_id}' resolves to '{canonical}'; "
            "switch to the configured model id."
        )
    if not isinstance(config.dimension, int):
        raise ValueError(
            f"Embedding model '{target_model_id}' has no dimension; cannot bind it."
        )
    return canonical, config.dimension, config


def _held_state(window: MigrationWindow) -> CollectionMigrationState:
    state = read_migration_state(window.collection_name)
    if state.state != FROZEN or (state.owner, state.updated_at) != (
        window.owner,
        window.updated_at,
    ):
        raise CollectionNotFrozenError(
            f"Collection '{window.collection_name}' is not frozen by this run; "
            "claim it first.",
            details={
                "collection": window.collection_name,
                "migration_state": state.state,
            },
        )
    return state


def switch_collection_embedding_model(
    window: MigrationWindow, target_model_id: str
) -> CollectionEmbeddingSwitch:
    """Bind the collection ``window`` holds to ``target_model_id``.

    The source is the binding read here, after the claim. The transition is
    recorded only while this run still holds the row version it wrote last, and
    the binding moves only if the collection row is unchanged since that read.

    Raises:
        CollectionNotFrozenError: If this run does not hold the window.
        EmbeddingAdapterError: If the model hub is unavailable or has no such model.
        ValueError: If the collection does not exist, the resolved id differs from
            the requested one, or the stored row moved while switching.
    """
    collection_name = window.collection_name
    state = _held_state(window)
    canonical_model_id, dimension, _ = _resolve_configured_model(target_model_id)

    stored = get_collection_sync(collection_name)
    observed_model_id = stored.embedding_model_id
    observed = _normalize_model_id(observed_model_id)
    source_model_id = observed
    if state.target_model_id == canonical_model_id and observed == canonical_model_id:
        # Re-applying a switch that already landed: keep the original rollback
        # origin, including an unbound one (``None``).
        source_model_id = state.source_model_id

    # The record goes first: if the binding write fails, retrying the switch is
    # idempotent and still knows the source model to roll back to.
    record_migration_transition(window, source_model_id, canonical_model_id)

    applied = write_collection_binding_sync(
        collection_name,
        canonical_model_id,
        dimension,
        # Exactly one binding guard: the store rejects both, and an unbound row
        # is not matched by an exact-value compare-and-set.
        expected_model_id=observed_model_id if observed is not None else None,
        require_uninitialized=observed is None,
        expected_updated_at=stored.updated_at,
    )
    if not applied:
        current = get_collection_sync(collection_name)
        # Both fields: a row bound to the target without a dimension is not switched.
        if (
            _normalize_model_id(current.embedding_model_id) != canonical_model_id
            or current.embedding_dimension != dimension
        ):
            raise ValueError(
                f"Collection '{collection_name}' changed while switching; retry "
                "the switch."
            )
        # The target is bound; accept that only while this run holds the window,
        # or a taken-over run would report the new owner's switch as its own.
        _held_state(window)

    logger.info(
        "Switched collection '%s' embedding model from '%s' to '%s'",
        collection_name,
        source_model_id,
        canonical_model_id,
    )
    return CollectionEmbeddingSwitch(
        collection_name=collection_name,
        source_model_id=source_model_id,
        target_model_id=canonical_model_id,
        target_dimension=dimension,
    )
