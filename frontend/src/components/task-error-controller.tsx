"use client"

// Reserved seam for surfacing coded terminal task failures (see
// `@/lib/task-error-events`). Stock xagent has no product-specific errors to
// present, so this is a no-op. An app-layer overlay can replace this file
// with a controller that listens for the `xagent:task-error` event and
// presents its own UI.
export function TaskErrorController(): null {
  return null
}
