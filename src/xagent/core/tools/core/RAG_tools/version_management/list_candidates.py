"""List candidates functionality for version management.

This module provides functionality for listing candidate versions
across different processing stages (parse, chunk, embed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from ..core.schemas import StepType

if TYPE_CHECKING:
    from ..kb import KBVersionCompatibilityFacade


def _get_version_compatibility_facade() -> "KBVersionCompatibilityFacade":
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().version_compatibility


def list_candidates(
    collection: str,
    doc_id: str,
    step_type: Union[StepType, str],
    model_tag: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = 50,
    order_by: str = "created_at desc",
) -> Dict[str, Any]:
    return _get_version_compatibility_facade().list_candidates(
        collection=collection,
        doc_id=doc_id,
        step_type=step_type,
        model_tag=model_tag,
        state=state,
        limit=limit,
        order_by=order_by,
    )
