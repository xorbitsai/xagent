"""JSON-serializable frame signatures used to validate approved actions.

A user approves one concrete action against one concrete frame. Between the
approval and the execution the page may change, so the approval must be
re-validated. Comparing raw screenshot bytes cannot do that job: a caret
blink, a carousel, a lazily loaded image, or an ad slot changes pixels without
changing what the approved action would do. These signatures capture the
structural facts the decision actually rested on, and they serialize so an
approval stays valid across a pause, a checkpoint, and a different process.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .policy import find_computer_target_element
from .schema import ComputerAction, ComputerElement, ComputerObservation


def element_signature(element: ComputerElement) -> dict[str, Any]:
    """Return the identity of one element as the policy understands it."""
    bounds = element.bounds
    return {
        "element_id": element.element_id,
        "label": element.label,
        "role": element.role,
        "text": element.text,
        "x": round(bounds.x, 4),
        "y": round(bounds.y, 4),
        "width": round(bounds.width, 4),
        "height": round(bounds.height, 4),
        "sensitive": bool(element.metadata.get("sensitive")),
        "focused": bool(element.metadata.get("focused")),
        "input_type": str(element.metadata.get("input_type") or ""),
    }


def structure_digest(observation: ComputerObservation) -> str:
    """Return a stable digest of the frame's interactive structure.

    Used for actions with no resolvable element (a drag, or a keypress with
    nothing focused): the approval was granted against a page layout, so the
    layout is what must still hold.
    """
    payload = json.dumps(
        [element_signature(element) for element in observation.elements],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def frame_signature(
    observation: ComputerObservation,
    actions: Sequence[ComputerAction],
) -> dict[str, Any]:
    """Capture what must still hold for an approved batch to remain valid."""
    return {
        "active_url": observation.active_url,
        "viewport": observation.viewport.model_dump(mode="json"),
        "structure": structure_digest(observation),
        "targets": [
            (
                element_signature(element)
                if (element := find_computer_target_element(action, observation))
                is not None
                else None
            )
            for action in actions
        ],
    }


def frame_signature_matches(
    expected: Mapping[str, Any] | None,
    observation: ComputerObservation,
    actions: Sequence[ComputerAction],
) -> bool:
    """Whether ``observation`` still satisfies a previously captured signature."""
    if not isinstance(expected, Mapping):
        return False
    current = frame_signature(observation, actions)
    if expected.get("active_url") != current["active_url"]:
        return False
    if expected.get("viewport") != current["viewport"]:
        return False

    expected_targets = expected.get("targets")
    if not isinstance(expected_targets, list) or len(expected_targets) != len(
        current["targets"]
    ):
        return False
    for expected_target, current_target in zip(expected_targets, current["targets"]):
        if (expected_target is None) != (current_target is None):
            return False
        if expected_target is None:
            # No element backs this action, so the surrounding layout is the
            # only evidence that the approval still describes the same page.
            if expected.get("structure") != current["structure"]:
                return False
            continue
        if dict(expected_target) != current_target:
            return False
    return True
