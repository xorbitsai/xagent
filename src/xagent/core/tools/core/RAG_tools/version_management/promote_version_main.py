"""Promote version main functionality for version management.

This module provides functionality for promoting candidate versions
to main versions with cascade cleanup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from ..core.exceptions import VersionManagementError
from ..core.schemas import StepType
from .cascade_cleaner import _cleanup_cascade_impl as cleanup_cascade

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..kb import KBVersionCompatibilityFacade


def _get_version_compatibility_facade() -> "KBVersionCompatibilityFacade":
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().version_compatibility


def _call_cleanup_cascade(
    collection: str,
    doc_id: str,
    step_type: StepType,
    technical_id: str,
    old_technical_id: Optional[str] = None,
    model_tag: Optional[str] = None,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, Any]:
    """
    Helper to call cleanup_cascade to compute deleted counts or execute cleanup.

    Args:
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type
        technical_id: Technical ID of the new main version
        old_technical_id: Technical ID of the old main version (if exists)
        model_tag: Model tag for embed step type
        preview_only: If True, only return preview without executing
        confirm: If True, execute the promotion

    Returns:
        Dictionary of deleted counts
    """
    if step_type == StepType.PARSE:
        return cleanup_cascade(
            collection=collection,
            doc_id=doc_id,
            scope="parse",
            new_parse_hash=technical_id,
            old_parse_hash=old_technical_id,
            preview_only=preview_only,
            confirm=confirm,
        )
    elif step_type == StepType.CHUNK:
        return cleanup_cascade(
            collection=collection,
            doc_id=doc_id,
            scope="chunk",
            new_parse_hash=technical_id,
            old_parse_hash=old_technical_id,
            preview_only=preview_only,
            confirm=confirm,
        )
    elif step_type == StepType.EMBED:
        if not model_tag:
            raise VersionManagementError("model_tag is required for embed step")
        return cleanup_cascade(
            collection=collection,
            doc_id=doc_id,
            scope="embeddings",
            model_tag=model_tag,
            preview_only=preview_only,
            confirm=confirm,
        )
    else:
        step_type_str = (
            step_type.value if isinstance(step_type, StepType) else str(step_type)
        )
        raise VersionManagementError(f"Invalid step_type: {step_type_str}")


def promote_version_main(
    collection: str,
    doc_id: str,
    step_type: Union[StepType, str],
    selected_id: str,
    operator: Optional[str] = None,
    preview_only: bool = False,
    confirm: bool = False,
    model_tag: Optional[str] = None,
) -> Dict[str, Any]:
    return _get_version_compatibility_facade().promote_version_main(
        collection=collection,
        doc_id=doc_id,
        step_type=step_type,
        selected_id=selected_id,
        operator=operator,
        preview_only=preview_only,
        confirm=confirm,
        model_tag=model_tag,
    )
