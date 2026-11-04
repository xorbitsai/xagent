import { apiRequest } from "@/lib/api-wrapper"
import { normalizeTaskStatus, type TaskStatus } from "@/lib/task-status"
import { getApiUrl } from "@/lib/utils"

export type AutoResumeCancelOutcome =
  | "cancelled"
  | "already_cancelled"
  | "resume_in_progress"
  | "not_pending"

export type AutoResumeCancelControlState =
  | "idle"
  | "running"
  | "pause_requested"
  | "paused"
  | "resume_requested"
  | "waiting_for_user"
  | "completed"
  | "failed"

export interface AutoResumeCancelSnapshot {
  taskId: number
  outcome: AutoResumeCancelOutcome
  status: TaskStatus
  controlState: AutoResumeCancelControlState
  runId: string | null
  stateVersion: number
  autoRecovery: Record<string, unknown> | null
  notice: Record<string, unknown> | null
}

export type CancelTaskAutoResumeFailure =
  | { kind: "transport" }
  | { kind: "malformed_response" }
  | { kind: "http"; status: number; code?: string }

export type CancelTaskAutoResumeResult =
  | { ok: true; snapshot: AutoResumeCancelSnapshot }
  | { ok: false; failure: CancelTaskAutoResumeFailure }

const OUTCOMES = new Set<AutoResumeCancelOutcome>([
  "cancelled",
  "already_cancelled",
  "resume_in_progress",
  "not_pending",
])

const CONTROL_STATES = new Set<AutoResumeCancelControlState>([
  "idle",
  "running",
  "pause_requested",
  "paused",
  "resume_requested",
  "waiting_for_user",
  "completed",
  "failed",
])

const isRecord = (value: unknown): value is Record<string, unknown> => (
  typeof value === "object" && value !== null && !Array.isArray(value)
)

const parseSnapshot = (value: unknown): AutoResumeCancelSnapshot | null => {
  if (!isRecord(value)) return null
  const status = normalizeTaskStatus(value.status)
  if (
    typeof value.task_id !== "number"
    || !Number.isInteger(value.task_id)
    || typeof value.outcome !== "string"
    || !OUTCOMES.has(value.outcome as AutoResumeCancelOutcome)
    || typeof value.control_state !== "string"
    || !CONTROL_STATES.has(value.control_state as AutoResumeCancelControlState)
    || !status
    || (value.run_id !== null && typeof value.run_id !== "string")
    || typeof value.state_version !== "number"
    || !Number.isInteger(value.state_version)
    || (value.auto_recovery !== null && !isRecord(value.auto_recovery))
    || (value.notice !== null && !isRecord(value.notice))
  ) return null

  return {
    taskId: value.task_id as number,
    outcome: value.outcome as AutoResumeCancelOutcome,
    status,
    controlState: value.control_state as AutoResumeCancelControlState,
    runId: value.run_id as string | null,
    stateVersion: value.state_version as number,
    autoRecovery: value.auto_recovery as Record<string, unknown> | null,
    notice: value.notice as Record<string, unknown> | null,
  }
}

export async function cancelTaskAutoResume(
  taskId: number,
  runId: string,
): Promise<CancelTaskAutoResumeResult> {
  let response: Response
  try {
    response = await apiRequest(
      `${getApiUrl()}/api/chat/task/${taskId}/auto-resume/cancel`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId }),
      },
    )
  } catch {
    return { ok: false, failure: { kind: "transport" } }
  }

  const body = await response.json().catch(() => null) as unknown
  if (!response.ok) {
    const detail = isRecord(body) && isRecord(body.detail) ? body.detail : null
    return {
      ok: false,
      failure: {
        kind: "http",
        status: response.status,
        ...(typeof detail?.code === "string" ? { code: detail.code } : {}),
      },
    }
  }

  const snapshot = parseSnapshot(body)
  return snapshot
    ? { ok: true, snapshot }
    : { ok: false, failure: { kind: "malformed_response" } }
}
