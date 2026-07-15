// Seam for surfacing coded terminal task failures to an app layer.
//
// Core stays quota/billing-agnostic: when a task ends with a machine-readable
// error_code (e.g. "quota_exceeded"), the chat context emits this event instead
// of hardcoding any product-specific copy or UI. Stock xagent mounts a no-op
// `TaskErrorController`, so nothing happens. An app layer (e.g. xagent-cloud)
// replaces that controller to present its own dialog — mirroring the backend
// quota-hook seam, where core forwards a code and the app layer decides what
// it means.
//
// The event fires only on the live terminal task_completed event, not on
// history replay after a reload — a reloaded failed task shows its reason in
// the transcript bubble, but the transient dialog is not re-raised.

export const TASK_ERROR_EVENT = "xagent:task-error"

export interface TaskErrorEventDetail {
  code: string
  details?: Record<string, unknown>
  // Backend's human-readable fallback message. May be an empty string when the
  // terminal event carried a code but no reason text; the app layer can show
  // its own localized copy instead and keep this as a fallback.
  message: string
  taskId: number | null
}

export function emitTaskError(detail: TaskErrorEventDetail): void {
  if (typeof window === "undefined") return
  window.dispatchEvent(new CustomEvent<TaskErrorEventDetail>(TASK_ERROR_EVENT, { detail }))
}
