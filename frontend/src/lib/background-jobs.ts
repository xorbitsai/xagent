import { apiRequest } from "@/lib/api-wrapper"

export interface BackgroundJobResponse {
  id: string
  user_id: number
  job_type: string
  queue: string
  status: string
  progress?: Record<string, unknown> | null
  result?: Record<string, unknown> | null
  error_message?: string | null
  celery_task_id?: string | null
  attempts: number
  max_attempts: number
  started_at?: string | null
  finished_at?: string | null
  created_at?: string | null
  updated_at?: string | null
}

interface BackgroundJobCapabilities {
  kb_ingest_mode?: string
}

export function isBackgroundJobResponse(value: unknown): value is BackgroundJobResponse {
  if (!value || typeof value !== "object") return false
  const candidate = value as Record<string, unknown>
  return (
    typeof candidate.id === "string" &&
    typeof candidate.job_type === "string" &&
    typeof candidate.queue === "string" &&
    typeof candidate.status === "string"
  )
}

export async function shouldUseBackgroundJobs(apiUrl: string): Promise<boolean> {
  try {
    const response = await apiRequest(`${apiUrl}/api/jobs/capabilities`)
    if (!response.ok) return false
    const data = await response.json() as BackgroundJobCapabilities
    return data.kb_ingest_mode === "celery"
  } catch {
    return false
  }
}

export function isBackgroundJobTerminal(job: BackgroundJobResponse): boolean {
  return ["succeeded", "failed", "cancelled"].includes(job.status)
}

export function getBackgroundJobProgressPercent(job: BackgroundJobResponse): number | null {
  if (job.status === "succeeded") return 100
  const completed = job.progress?.completed
  const total = job.progress?.total
  if (typeof completed === "number" && typeof total === "number" && total > 0) {
    return Math.max(0, Math.min(100, (completed / total) * 100))
  }
  return null
}

export function getBackgroundJobProgressMessage(job: BackgroundJobResponse): string | null {
  const message = job.progress?.message
  if (typeof message === "string" && message.trim()) return message
  return job.error_message || null
}

export function getBackgroundJobResult(job: BackgroundJobResponse): Record<string, unknown> | null {
  return job.result && typeof job.result === "object" ? job.result : null
}

export function getBackgroundJobFailureMessage(
  job: BackgroundJobResponse,
  fallbackMessage: string
): string {
  const resultMessage = job.result?.message
  if (typeof resultMessage === "string" && resultMessage.trim()) {
    return resultMessage
  }
  return job.error_message || fallbackMessage
}

export async function waitForBackgroundJob(
  apiUrl: string,
  initialJob: BackgroundJobResponse,
  onUpdate?: (job: BackgroundJobResponse) => void
): Promise<BackgroundJobResponse> {
  let job = initialJob
  onUpdate?.(job)

  while (!isBackgroundJobTerminal(job)) {
    await new Promise(resolve => window.setTimeout(resolve, 1000))
    const response = await apiRequest(`${apiUrl}/api/jobs/${job.id}`)
    if (!response.ok) {
      throw new Error(`Failed to fetch background job ${job.id}`)
    }
    const data = await response.json()
    if (!isBackgroundJobResponse(data)) {
      throw new Error(`Invalid background job response for ${job.id}`)
    }
    job = data
    onUpdate?.(job)
  }

  return job
}
