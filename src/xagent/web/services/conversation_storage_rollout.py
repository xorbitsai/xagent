"""Default-off switch deciding which new tasks use V2 execution-event storage.

``Task.conversation_storage_version`` is 1 (legacy conversation storage) or 2
(execution-event storage). This module answers one question: which version a
*new* task gets.

Operating rules
    * Default off: with no configuration every new task is V1.
    * The version is decided once, at INSERT, by the creation site calling
      ``new_task_conversation_storage_version``. Every later read or write
      (follow-ups, resume, recovery, auto-resume) uses the version stored on
      the task row, never process configuration.
    * Turning an entry off affects only tasks created afterwards. Existing V2
      tasks keep reading and writing V2 and may still be continued, so the
      version is never rewritten and no process may be rolled back below the
      minimum compatible release.
    * Runtime telemetry must stay enabled on every process once V2 tasks
      exist; ``report_unobserved_execution_events`` flags processes that
      enable V2 entries while telemetry export is not running.

Configuration (read from the environment on every call; there is no frozen
singleton, so tests and creation sites need no initialization)
    ``XAGENT_EXECUTION_EVENT_STORAGE_ENTRIES``: comma-separated entry names
    from ``STORAGE_ENTRIES``. Empty means all V1.
    ``XAGENT_EXECUTION_EVENT_STORAGE_USER_IDS``: optional comma-separated
    positive owner user ids. Empty means no per-user restriction.

Unknown entries, duplicate entries and invalid ids fail process startup via
``validate_conversation_storage_rollout_at_startup``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ...config import (
    get_execution_event_storage_entries,
    get_execution_event_storage_user_ids,
)
from .ops_signals import EXECUTION_EVENTS_UNOBSERVED, register_degradation

logger = logging.getLogger(__name__)

STORAGE_ENTRIES = frozenset(
    {
        "web_chat",
        "agent_preview",
        "sdk",
        "a2a",
        "widget",
        "shared_link",
        "channel",
        "trigger",
        "workforce",
    }
)


class ConversationStorageConfigError(ValueError):
    """The execution-event storage rollout configuration is invalid."""


@dataclass(frozen=True)
class ConversationStoragePolicy:
    entries: frozenset[str]
    # None means no per-user restriction.
    owner_user_ids: frozenset[int] | None


def load_conversation_storage_policy() -> ConversationStoragePolicy:
    """Read and validate the rollout configuration from the environment."""
    entries: list[str] = []
    seen: dict[str, int] = {}
    for position, entry in enumerate(get_execution_event_storage_entries()):
        if entry not in STORAGE_ENTRIES:
            raise ConversationStorageConfigError(
                f"Unknown execution event storage entry {entry!r}; "
                f"allowed entries: {sorted(STORAGE_ENTRIES)}"
            )
        if entry in seen:
            raise ConversationStorageConfigError(
                f"Duplicate execution event storage entry {entry!r} "
                f"(first listed at position {seen[entry] + 1})"
            )
        seen[entry] = position
        entries.append(entry)

    raw_ids = get_execution_event_storage_user_ids()
    owner_user_ids: frozenset[int] | None = None
    if raw_ids:
        parsed: set[int] = set()
        for raw in raw_ids:
            try:
                value = int(raw)
            except ValueError:
                raise ConversationStorageConfigError(
                    f"Invalid execution event storage user id {raw!r}: "
                    "must be a positive integer"
                ) from None
            if value <= 0:
                raise ConversationStorageConfigError(
                    f"Invalid execution event storage user id {raw!r}: "
                    "must be a positive integer"
                )
            parsed.add(value)
        owner_user_ids = frozenset(parsed)

    return ConversationStoragePolicy(
        entries=frozenset(entries), owner_user_ids=owner_user_ids
    )


def validate_conversation_storage_rollout_at_startup() -> ConversationStoragePolicy:
    """Fail fast on bad configuration and log the resolved policy once."""
    policy = load_conversation_storage_policy()
    logger.info(
        "Conversation storage rollout configured: entries=%s owner_user_ids=%s",
        sorted(policy.entries),
        "all" if policy.owner_user_ids is None else len(policy.owner_user_ids),
    )
    return policy


def new_task_conversation_storage_version(
    *, entry: str, owner_user_id: int, via: str | None = None
) -> int:
    """Return the storage version (1 or 2) for a task about to be inserted.

    ``entry`` names the creation site. ``via`` is the initiating entry for
    composite creations (a workforce run started from a widget); when given it
    must be enabled too.
    """
    if entry not in STORAGE_ENTRIES:
        raise ValueError(f"Unknown execution event storage entry: {entry!r}")
    if via is not None and via not in STORAGE_ENTRIES:
        raise ValueError(f"Unknown execution event storage entry: {via!r}")

    policy = load_conversation_storage_policy()
    if entry not in policy.entries:
        return 1
    if via is not None and via not in policy.entries:
        return 1
    if policy.owner_user_ids is not None and owner_user_id not in policy.owner_user_ids:
        return 1
    return 2


def report_unobserved_execution_events(*, telemetry_started: bool) -> None:
    """Flag V2 entries enabled while runtime telemetry export is not running.

    Independent of the user allowlist: any enabled entry may produce V2 tasks.
    The log carries entry names only, never endpoints, secrets or payloads.
    """
    if telemetry_started:
        return
    entries = sorted(load_conversation_storage_policy().entries)
    if not entries:
        return
    logger.warning(
        "Execution event storage V2 is enabled for new tasks (entries=%s) "
        "but runtime telemetry export is not running",
        entries,
    )
    register_degradation(EXECUTION_EVENTS_UNOBSERVED, f"entries={entries}")
