"""Shared persisted trace event type names used across producers and consumers."""

LEGACY_GENERAL_ERROR_EVENT_TYPE = "trace_error"
TASK_GENERAL_ERROR_EVENT_TYPE = "task_error_general"
STEP_GENERAL_ERROR_EVENT_TYPE = "step_error_general"

GENERAL_ERROR_EVENT_TYPES = frozenset(
    {
        LEGACY_GENERAL_ERROR_EVENT_TYPE,
        TASK_GENERAL_ERROR_EVENT_TYPE,
        STEP_GENERAL_ERROR_EVENT_TYPE,
    }
)
