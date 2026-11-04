"""Merge-insert error classification for the vector store data plane.

``merge_insert`` upserts can fail for two very different reasons:

* **Non-recoverable** schema / type / dimension problems — retrying with
  ``add()`` would only corrupt the table, so these must re-raise immediately.
* **Recoverable** transient problems — falling back to ``add()`` is safe.

This leaf module hosts the classifier so the LanceDB store can decide locally
(its only live consumer is :meth:`LanceDBVectorIndexStore.upsert_embeddings`).
Keeping it in ``storage/`` removes the historical upward import from the
``vector_storage`` layer.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def is_non_recoverable_merge_error(error: Exception) -> bool:
    """Classify a ``merge_insert`` failure as recoverable or non-recoverable.

    Returns ``True`` when the error is non-recoverable (re-raise without
    fallback to ``add()``), ``False`` otherwise.
    """
    # Built-in Python exceptions that always indicate non-recoverable issues,
    # regardless of the installed LanceDB version.
    if isinstance(error, (AttributeError, TypeError, ValueError)):
        return True

    # Explicit LanceDB exception types when available.
    try:  # pragma: no cover - depends on installed lancedb version
        from lancedb.exceptions import (  # type: ignore[import-not-found]
            LanceDBSchemaError,
            LanceDBValidationError,
        )

        if isinstance(error, (LanceDBSchemaError, LanceDBValidationError)):
            return True
        # Known LanceDB exception type but not schema/validation -> recoverable.
        return False
    except Exception:  # noqa: BLE001
        # LanceDB exception types not available - fall through to string match.
        pass

    # String-matching fallback for older LanceDB versions.
    error_str = str(error).lower()
    non_recoverable_keywords = (
        "schema",
        "type mismatch",
        "type error",
        "validation",
        "dimension",
        "field",
        "column",
    )
    is_non_recoverable = any(
        keyword in error_str for keyword in non_recoverable_keywords
    )

    if is_non_recoverable:
        logger.warning(
            "Error classified as non-recoverable via string matching. "
            "Upgrade LanceDB to get accurate exception-based classification. "
            "Error: %s",
            error,
        )
    else:
        logger.debug(
            "Error classified as recoverable via string matching (no schema "
            "keywords found). Attempting fallback to add() method. Error: %s",
            error,
        )

    return is_non_recoverable
