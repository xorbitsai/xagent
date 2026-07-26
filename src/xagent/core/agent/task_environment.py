from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

TASK_ENVIRONMENT_METADATA_KEY = "task_environment"

_COMPUTER_TARGETS: dict[str, dict[str, Any]] = {
    "extension_relay": {
        "runtime_kind": "extension_relay",
        "target_kind": "browser",
        "display_name": "My browser",
        "scope": (
            "the single browser tab explicitly approved by the user through "
            "the Xagent browser extension"
        ),
        "preferred_input_modalities": ["image"],
    },
    "desktop_relay": {
        "runtime_kind": "desktop_relay",
        "target_kind": "desktop",
        "display_name": "My computer",
        "scope": (
            "the single desktop window explicitly approved by the user through "
            "Xagent Desktop Relay"
        ),
        "preferred_input_modalities": ["image"],
    },
}


def build_task_environment(
    *,
    computer_runtime_kind: str | None,
) -> dict[str, Any]:
    """Build immutable task-resource semantics from the locked task selection."""

    target = _COMPUTER_TARGETS.get(str(computer_runtime_kind or "").strip())
    if target is None:
        return {}
    return {"computer": deepcopy(target)}


def normalize_task_environment(value: Any) -> dict[str, Any]:
    """Return a detached JSON-compatible task environment."""

    if not isinstance(value, Mapping):
        return {}
    computer = value.get("computer")
    if not isinstance(computer, Mapping):
        return {}

    runtime_kind = str(computer.get("runtime_kind") or "").strip()
    canonical = _COMPUTER_TARGETS.get(runtime_kind)
    if canonical is None:
        return {}
    return {"computer": deepcopy(canonical)}


def task_environment_system_context(metadata: Mapping[str, Any]) -> str:
    environment = normalize_task_environment(
        metadata.get(TASK_ENVIRONMENT_METADATA_KEY)
    )
    computer = environment.get("computer")
    if not isinstance(computer, dict):
        return ""

    display_name = computer["display_name"]
    target_kind = computer["target_kind"]
    scope = computer["scope"]
    return (
        "Selected computer target:\n"
        f"- Target: {display_name} ({target_kind}).\n"
        f"- Scope: {scope}.\n"
        "- This selection makes the target available to the computer tool; it "
        "does not mean that a screenshot has already been captured.\n"
        "- When the user refers to the current, visible, selected, or authorized "
        "page or window, inspect this target with the computer tool before "
        "searching workspace files or guessing its contents.\n"
        "- Use the computer tool only when it is relevant to the request; the "
        "selection does not require computer access for unrelated work."
    )


def task_environment_input_modalities(
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    environment = normalize_task_environment(
        metadata.get(TASK_ENVIRONMENT_METADATA_KEY)
    )
    computer = environment.get("computer")
    if not isinstance(computer, dict):
        return ()
    values = computer.get("preferred_input_modalities")
    if not isinstance(values, list):
        return ()
    return tuple(
        dict.fromkeys(
            str(value).strip().lower() for value in values if str(value).strip()
        )
    )
