import { isJsonRecord } from "./api-wrapper"

export interface TaskCreateCore {
  taskId: number
  title: string
  status: string
  createdAt: string
}

export function normalizeTaskPromptTitle(prompt: string): string {
  return prompt.trim().replace(/\s+/gu, " ")
}

export function parseTaskCreateCore(value: unknown): TaskCreateCore | null {
  if (!isJsonRecord(value)) return null
  const { task_id, title, status, created_at } = value
  if (typeof task_id !== "number" || !Number.isSafeInteger(task_id) || task_id <= 0) return null
  if (typeof title !== "string" || typeof status !== "string" || typeof created_at !== "string") return null
  return { taskId: task_id, title, status, createdAt: created_at }
}
