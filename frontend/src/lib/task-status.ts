export type TaskStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "paused"
  | "waiting_for_user"

const TASK_STATUSES = new Set<TaskStatus>([
  "pending",
  "running",
  "completed",
  "failed",
  "paused",
  "waiting_for_user",
])

const STOPPED_TASK_STATUSES = new Set<TaskStatus>([
  "completed",
  "failed",
  "paused",
  "waiting_for_user",
])

const PAUSABLE_TASK_STATUSES = new Set<TaskStatus>([
  "pending",
  "running",
])

export const normalizeTaskStatus = (status: unknown): TaskStatus | undefined => {
  if (typeof status !== "string") return undefined

  const normalized = status.trim().toLowerCase()
  return TASK_STATUSES.has(normalized as TaskStatus)
    ? (normalized as TaskStatus)
    : undefined
}

export const isStoppedTaskStatus = (status: unknown): boolean => {
  const normalized = normalizeTaskStatus(status)
  return normalized ? STOPPED_TASK_STATUSES.has(normalized) : false
}

export const isPausableTaskStatus = (status: unknown): boolean => {
  const normalized = normalizeTaskStatus(status)
  return normalized ? PAUSABLE_TASK_STATUSES.has(normalized) : false
}
