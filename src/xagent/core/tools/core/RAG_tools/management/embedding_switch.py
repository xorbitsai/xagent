"""Switch one collection's bound embedding model while it is frozen.

The switch writes the model id and dimension together, records the transition in
``collection_migration_state``, and leaves every other metadata column and
``main_pointers`` untouched. Rolling back is the same call with the previous
model, after the collection is frozen again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..core.exceptions import CollectionNotFrozenError
from ..utils.model_resolver import resolve_embedding_adapter
from .collection_manager import (
    _normalize_model_id,
    get_collection_sync,
    write_collection_binding_sync,
)
from .migration_state import (
    FROZEN,
    CollectionMigrationState,
    read_migration_state,
    write_migration_state,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectionEmbeddingSwitch:
    """One applied switch: what the collection used and what it uses now."""

    collection_name: str
    source_model_id: Optional[str]
    target_model_id: str
    target_dimension: int


def switch_collection_embedding_model(
    collection_name: str, target_model_id: str
) -> CollectionEmbeddingSwitch:
    """Bind ``collection_name`` to ``target_model_id``.

    Raises:
        CollectionNotFrozenError: If the collection is not frozen for migration.
        ValueError: If the collection does not exist or the model has no dimension.
    """
    state = read_migration_state(collection_name)
    if state.state != FROZEN:
        raise CollectionNotFrozenError(
            f"Collection '{collection_name}' is not frozen for an embedding "
            "switch; freeze it first.",
            details={"collection": collection_name, "migration_state": state.state},
        )

    embedding_config, _ = resolve_embedding_adapter(target_model_id)
    canonical_model_id = (
        _normalize_model_id(getattr(embedding_config, "id", None)) or target_model_id
    )
    dimension = getattr(embedding_config, "dimension", None)
    if not isinstance(dimension, int):
        raise ValueError(
            f"Embedding model '{target_model_id}' has no dimension; cannot bind it."
        )

    current = get_collection_sync(collection_name)
    source_model_id = _normalize_model_id(current.embedding_model_id)
    if (
        state.target_model_id == canonical_model_id
        and source_model_id == canonical_model_id
        and state.source_model_id
    ):
        # Re-applying a switch that already landed: keep the original rollback point.
        source_model_id = state.source_model_id

    # The record goes first: if the binding write fails, retrying the switch is
    # idempotent and still knows the source model to roll back to.
    write_migration_state(
        collection_name,
        CollectionMigrationState(
            state=FROZEN,
            source_model_id=source_model_id,
            target_model_id=canonical_model_id,
        ),
    )
    write_collection_binding_sync(collection_name, canonical_model_id, dimension)

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
