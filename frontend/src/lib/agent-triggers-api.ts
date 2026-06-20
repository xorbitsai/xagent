"use client"

import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"

export type AgentTriggerType = "webhook" | "scheduled"
export type AgentTriggerRunStatus = "pending" | "running" | "completed" | "failed"

export interface AgentTrigger {
  id: number
  user_id: number
  agent_id: number
  type: AgentTriggerType
  name: string
  enabled: boolean
  config: Record<string, unknown>
  prompt_template: string | null
  webhook_token: string | null
  webhook_secret?: string | null
  next_run_at: string | null
  last_run_at: string | null
  last_error: string | null
  created_at: string | null
  updated_at: string | null
}

export interface AgentTriggerRun {
  id: number
  trigger_id: number
  task_id: number | null
  background_job_id: string | null
  status: AgentTriggerRunStatus
  source_event_id: string | null
  payload_snapshot: Record<string, unknown> | null
  idempotency_key: string
  error_message: string | null
  started_at: string | null
  finished_at: string | null
  created_at: string | null
  updated_at: string | null
}

export interface AgentTriggerPayload {
  type?: AgentTriggerType
  name?: string | null
  enabled?: boolean
  config?: Record<string, unknown>
  prompt_template?: string | null
  secret?: string | null
  rotate_secret?: boolean
}

export interface AgentTriggerTestPayload {
  payload: Record<string, unknown>
  source_event_id?: string | null
}

export interface AgentTriggerTestResponse {
  trigger_run: AgentTriggerRun
  duplicate: boolean
}

function jsonHeaders(): HeadersInit {
  return {
    "Content-Type": "application/json",
  }
}

function triggerUrl(agentId: number | string, triggerId?: number | string): string {
  const base = `${getApiUrl()}/api/agents/${agentId}/triggers`
  return triggerId === undefined ? base : `${base}/${triggerId}`
}

function formatApiDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail.trim()) {
    return detail
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String(item.msg)
        }
        return null
      })
      .filter(Boolean)
    if (messages.length > 0) {
      return messages.join("; ")
    }
  }
  return fallback
}

async function parseApiError(response: Response, fallback: string): Promise<Error> {
  try {
    const data = await response.json()
    return new Error(formatApiDetail(data?.detail, fallback))
  } catch {
    return new Error(fallback)
  }
}

export async function listAgentTriggers(
  agentId: number | string,
): Promise<AgentTrigger[]> {
  const response = await apiRequest(triggerUrl(agentId))
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load triggers")
  }
  return response.json()
}

export async function createAgentTrigger(
  agentId: number | string,
  payload: AgentTriggerPayload & { type: AgentTriggerType },
): Promise<AgentTrigger> {
  const response = await apiRequest(triggerUrl(agentId), {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to create trigger")
  }
  return response.json()
}

export async function updateAgentTrigger(
  agentId: number | string,
  triggerId: number | string,
  payload: AgentTriggerPayload,
): Promise<AgentTrigger> {
  const response = await apiRequest(triggerUrl(agentId, triggerId), {
    method: "PATCH",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to update trigger")
  }
  return response.json()
}

export async function deleteAgentTrigger(
  agentId: number | string,
  triggerId: number | string,
): Promise<void> {
  const response = await apiRequest(triggerUrl(agentId, triggerId), {
    method: "DELETE",
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to delete trigger")
  }
}

export async function listAgentTriggerRuns(
  agentId: number | string,
  triggerId: number | string,
): Promise<AgentTriggerRun[]> {
  const response = await apiRequest(`${triggerUrl(agentId, triggerId)}/runs`)
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load trigger runs")
  }
  return response.json()
}

export async function testAgentTrigger(
  agentId: number | string,
  triggerId: number | string,
  payload: AgentTriggerTestPayload,
): Promise<AgentTriggerTestResponse> {
  const response = await apiRequest(`${triggerUrl(agentId, triggerId)}/test`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to test trigger")
  }
  return response.json()
}
