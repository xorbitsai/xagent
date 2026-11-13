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
    read_migration_state,
    update_migration_state_if_frozen,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectionEmbeddingSwitch:
    """One applied switch: what the collection used and what it uses now."""

    collection_name: str
    source_model_id: Optional[str]
    target_model_id: str
    target_dimension: int


def resolve_configured_model(target_model_id: str) -> tuple[str, int]:
    """Public alias: migrations resolve their target through this one rule."""
    return _resolve_configured_model(target_model_id)


def _resolve_configured_model(target_model_id: str) -> tuple[str, int]:
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
    return canonical, config.dimension


def switch_collection_embedding_model(
    collection_name: str, target_model_id: str
) -> CollectionEmbeddingSwitch:
    """Bind ``collection_name`` to ``target_model_id``.

    Raises:
        CollectionNotFrozenError: If the collection is not frozen for migration.
        EmbeddingAdapterError: If the model hub is unavailable or has no such model.
        ValueError: If the collection does not exist, the resolved id differs from
            the requested one, or the stored binding moved while switching.
    """
    state = read_migration_state(collection_name)
    if state.state != FROZEN:
        raise CollectionNotFrozenError(
            f"Collection '{collection_name}' is not frozen for an embedding "
            "switch; freeze it first.",
            details={"collection": collection_name, "migration_state": state.state},
        )

    canonical_model_id, dimension = _resolve_configured_model(target_model_id)

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
    if not update_migration_state_if_frozen(
        collection_name,
        CollectionMigrationState(
            state=FROZEN,
            source_model_id=source_model_id,
            target_model_id=canonical_model_id,
        ),
    ):
        raise CollectionNotFrozenError(
            f"Collection '{collection_name}' was unfrozen while switching.",
            details={"collection": collection_name, "migration_state": "unfrozen"},
        )

    applied = write_collection_binding_sync(
        collection_name,
        canonical_model_id,
        dimension,
        # Exactly one guard: the store rejects both, and an unbound row is not
        # matched by an exact-value compare-and-set.
        expected_model_id=observed_model_id if observed is not None else None,
        require_uninitialized=observed is None,
    )
    if not applied:
        # A concurrent identical switch may have written the same target.
        current_model_id = _normalize_model_id(
            get_collection_sync(collection_name).embedding_model_id
        )
        if current_model_id != canonical_model_id:
            raise ValueError(
                f"Collection '{collection_name}' embedding model changed while "
                "switching; retry the switch."
            )

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
