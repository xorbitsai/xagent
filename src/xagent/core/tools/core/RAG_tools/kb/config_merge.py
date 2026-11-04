"""Merge rule for the collection config row, shared by every ingest entry point."""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _dumps(settings: dict) -> str:
    """Match ``model_dump_json``'s compact separators."""
    return json.dumps(settings, separators=(",", ":"))


def merge_collection_config_json(
    existing_config_json: Optional[str],
    new_config_json: str,
) -> str:
    """Keep settings the finished ingest did not set, e.g. a rerank binding.

    The config row is replaced wholesale, and the write lands after the ingest
    rather than before it, so a key the user saved while the ingest was running
    would otherwise be dropped. The protection is narrower than it looks: the
    web and API routes fill in the whole chunking group, so those keys are always
    present in ``new_config_json`` and the finished run always wins them. Only
    keys the caller left unset — a binding owned by another feature, or the
    chunking group on the agent path, which sets just the embedding model —
    survive from ``existing_config_json``.

    Every merge result is serialized here, in the same compact form pydantic
    writes, so the stored text does not depend on which branch produced it. The
    two guard clauses below are the exception: unparsable or non-object input is
    handed back untouched rather than reshaped.
    """
    try:
        new_settings = json.loads(new_config_json)
    except (TypeError, ValueError):
        return new_config_json

    if not isinstance(new_settings, dict):
        return new_config_json

    if not existing_config_json:
        return _dumps(new_settings)

    try:
        existing_settings = json.loads(existing_config_json)
    except (TypeError, ValueError) as exc:
        logger.warning("Unreadable existing collection config, replacing it: %s", exc)
        return _dumps(new_settings)

    if not isinstance(existing_settings, dict):
        return _dumps(new_settings)

    return _dumps({**existing_settings, **new_settings})
