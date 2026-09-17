"""Promote version main functionality for version management.

This module provides functionality for promoting candidate versions
to main versions with cascade cleanup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from ..core.schemas import StepType

if TYPE_CHECKING:
    from ..kb import KBVersionCompatibilityFacade


def _get_version_compatibility_facade() -> "KBVersionCompatibilityFacade":
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().version_compatibility


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
